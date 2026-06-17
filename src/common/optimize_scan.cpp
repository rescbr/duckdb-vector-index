#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/statistics/node_statistics.hpp"
#include "duckdb/storage/table/data_table_info.hpp"
#include "duckdb/storage/table/table_index_list.hpp"

#include "vindex/label_filter.hpp"
#include "vindex/vector_index.hpp"
#include "vindex/vector_index_registry.hpp"
#include "vindex/vector_index_scan.hpp"

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// Label-filter extraction helpers
// ---------------------------------------------------------------------------

// Convert a single ConstantFilter on a BIGINT label column to a partial
// LabelFilter. Returns LabelFilter::None() if the filter type is not a
// supported comparison.
static LabelFilter ConstantFilterToLabel(const ConstantFilter &cf) {
	const auto &val = cf.constant;
	if (val.type().id() != LogicalTypeId::BIGINT && val.type().id() != LogicalTypeId::INTEGER &&
	    val.type().id() != LogicalTypeId::TINYINT && val.type().id() != LogicalTypeId::SMALLINT) {
		return LabelFilter::None();
	}
	const int64_t v = val.GetValue<int64_t>();
	switch (cf.comparison_type) {
		case ExpressionType::COMPARE_EQUAL:
			return LabelFilter::Equals(v);
		case ExpressionType::COMPARE_GREATERTHAN:
			return LabelFilter::Range(v == INT64_MAX ? INT64_MAX : v + 1, INT64_MAX);
		case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
			return LabelFilter::Range(v, INT64_MAX);
		case ExpressionType::COMPARE_LESSTHAN:
			return LabelFilter::Range(INT64_MIN, v == INT64_MIN ? INT64_MIN : v - 1);
		case ExpressionType::COMPARE_LESSTHANOREQUALTO:
			return LabelFilter::Range(INT64_MIN, v);
		default:
			return LabelFilter::None();
	}
}

// Fold multiple partial LabelFilters into one by intersection.
// Equality dominates: if any partial is EQUALS(v), and v satisfies all
// other partials, the result is EQUALS(v). Otherwise unsatisfiable.
static LabelFilter FoldLabelFilters(const vector<LabelFilter> &partials) {
	if (partials.size() == 1) {
		return partials[0];
	}
	for (const auto &p : partials) {
		if (p.kind == LabelFilter::Kind::EQUALS) {
			for (const auto &q : partials) {
				if (&p == &q) {
					continue;
				}
				if (!q.Matches(p.value)) {
					return LabelFilter::None();
				}
			}
			return p;
		}
	}
	int64_t lo = INT64_MIN, hi = INT64_MAX;
	for (const auto &p : partials) {
		if (p.kind != LabelFilter::Kind::RANGE) {
			continue;
		}
		lo = std::max(lo, p.lo);
		hi = std::min(hi, p.hi);
	}
	if (lo > hi) {
		return LabelFilter::None();
	}
	return LabelFilter::Range(lo, hi);
}

// Recursively collect partial LabelFilters from a TableFilter tree.
static void CollectLabelPartials(const TableFilter &filter, vector<LabelFilter> &out) {
	switch (filter.filter_type) {
		case TableFilterType::CONSTANT_COMPARISON: {
			auto &cf = filter.Cast<ConstantFilter>();
			auto partial = ConstantFilterToLabel(cf);
			if (partial.IsActive()) {
				out.push_back(partial);
			}
			break;
		}
		case TableFilterType::CONJUNCTION_AND: {
			auto &conj = filter.Cast<ConjunctionAndFilter>();
			for (auto &child : conj.child_filters) {
				CollectLabelPartials(*child, out);
			}
			break;
		}
		default:
			break;
	}
}

// Try to extract a LabelFilter from the table filters on the designated
// label column. Returns LabelFilter::None() if no label predicates are found
// or the index does not support label filtering.
static LabelFilter TryExtractLabelFilter(const VectorIndex &vi, LogicalGet &get) {
	if (!vi.SupportsLabelFilter()) {
		return LabelFilter::None();
	}
	const auto &label_col = vi.GetLabelColumn();
	if (label_col.empty()) {
		return LabelFilter::None();
	}

	// Resolve label column name → logical column ID by scanning the table's
	// column definitions.
	const auto &duck_table = get.GetTable()->Cast<DuckTableEntry>();
	const auto &columns = duck_table.GetColumns();
	idx_t label_logical_id = DConstants::INVALID_INDEX;
	for (const auto &col : columns.Logical()) {
		if (StringUtil::CIEquals(col.GetName(), label_col)) {
			label_logical_id = col.Oid();
			break;
		}
	}
	if (label_logical_id == DConstants::INVALID_INDEX) {
		return LabelFilter::None();
	}

	// Search the table filters for the label column.
	for (const auto &entry : get.table_filters.filters) {
		const idx_t filter_col_id = entry.first;
		if (filter_col_id != label_logical_id) {
			continue;
		}
		vector<LabelFilter> partials;
		CollectLabelPartials(*entry.second, partials);
		if (partials.empty()) {
			return LabelFilter::None();
		}
		return FoldLabelFilters(partials);
	}
	return LabelFilter::None();
}

// ---------------------------------------------------------------------------
// TopN → index scan rewrite
//
// Registry-driven: iterates every registered VectorIndex type name and asks
// whether it can service the projection's distance expression. The bind data
// and table function are the generic VectorIndexScan{BindData,Function}, so
// adding a new algorithm only requires registering its TYPE_NAME and
// subclassing VectorIndex. No changes here.
// ---------------------------------------------------------------------------

class VectorIndexScanOptimizer : public OptimizerExtension {
public:
	VectorIndexScanOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		auto &op = *plan;
		if (op.type != LogicalOperatorType::LOGICAL_TOP_N) {
			return false;
		}
		auto &top_n = op.Cast<LogicalTopN>();
		if (top_n.orders.size() != 1) {
			return false;
		}
		const auto &order = top_n.orders[0];
		if (order.type != OrderType::ASCENDING) {
			return false;
		}
		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			return false;
		}
		const auto &bound_column_ref = order.expression->Cast<BoundColumnRefExpression>();

		if (top_n.children.size() != 1 || top_n.children.front()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			return false;
		}

		auto &projection = top_n.children.front()->Cast<LogicalProjection>();
		const auto projection_index = bound_column_ref.binding.column_index;
		const auto &projection_expr = projection.expressions[projection_index];

		if (projection.children.size() != 1 || projection.children.front()->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get_ptr = projection.children.front();
		auto &get = get_ptr->Cast<LogicalGet>();
		if (get.function.name != "seq_scan") {
			return false;
		}
		if (get.dynamic_filters && get.dynamic_filters->HasFilters()) {
			return false;
		}

		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		unique_ptr<VectorIndexScanBindData> bind_data;
		vector<reference<Expression>> bindings;

		// Iterate all registered vector-index algorithms.
		for (const auto &type_name : VectorIndexRegistry::Instance().TypeNames()) {
			table_info.BindIndexes(context, type_name.c_str());
		}

		for (auto &index : table_info.GetIndexes().Indexes()) {
			if (!index.IsBound()) {
				continue;
			}
			auto *vi = VectorIndexRegistry::TryCast(index.Cast<BoundIndex>());
			if (!vi) {
				continue;
			}

			bindings.clear();
			if (!vi->TryMatchDistanceFunction(projection_expr, bindings)) {
				continue;
			}
			unique_ptr<Expression> index_expr;
			if (!vi->TryBindIndexExpression(get, index_expr)) {
				continue;
			}

			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];
			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					continue;
				}
			}

			const auto vector_size = vi->GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}

			// Over-fetch by `rerank_multiple`. Index returns candidates; the
			// surviving LogicalTopN above does the exact-distance final sort.
			// The scan dispatches through the VectorIndex virtual surface, so
			// we hand it the BoundIndex unchanged.
			const idx_t rerank = vi->GetRerankMultiple(context);
			const idx_t cand_limit = top_n.limit * rerank;
			bind_data =
			    make_uniq<VectorIndexScanBindData>(duck_table, index, cand_limit, std::move(query_vector));
			// Extract label-column filter predicates if the index supports it.
			bind_data->label_filter = TryExtractLabelFilter(*vi, get);
			break;
		}

		if (!bind_data) {
			return false;
		}

		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.function = VectorIndexScanFunction::GetFunction();
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		// NOTE: We intentionally do NOT remove `top_n` from the plan. The index
		// scan now returns candidates (k × rerank_multiple), and the TopN runs
		// the exact-distance ORDER BY + LIMIT over them. This is the clean
		// invariant we want even when rerank_multiple == 1 (TopN over exactly
		// k rows is a no-op but keeps the plan shape uniform).

		// Pull filters up above the index scan (it does not support pushdown).
		get.projection_ids.clear();
		get.types.clear();

		auto new_filter = make_uniq<LogicalFilter>();
		auto &column_ids = get.GetColumnIds();
		for (const auto &entry : get.table_filters.filters) {
			idx_t column_id = entry.first;
			auto &type = get.returned_types[column_id];
			bool found = false;
			for (idx_t i = 0; i < column_ids.size(); i++) {
				if (column_ids[i].GetPrimaryIndex() == column_id) {
					column_id = i;
					found = true;
					break;
				}
			}
			if (!found) {
				throw InternalException("Could not find column id for filter");
			}
			auto column = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(get.table_index, column_id));
			new_filter->expressions.push_back(entry.second->ToExpression(*column));
		}
		new_filter->children.push_back(std::move(get_ptr));
		new_filter->ResolveOperatorTypes();
		get_ptr = std::move(new_filter);

		// TopN stays in the plan — it performs exact-distance rerank + final LIMIT
		// over the (k × rerank_multiple) candidates the index scan emits.
		return true;
	}

	static bool OptimizeChildren(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		auto ok = TryOptimize(context, plan);
		for (auto &child : plan->children) {
			ok |= OptimizeChildren(context, child);
		}
		return ok;
	}

	static void MergeProjections(unique_ptr<LogicalOperator> &plan) {
		if (plan->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			if (plan->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &child = plan->children[0];
				if (child->children[0]->type == LogicalOperatorType::LOGICAL_GET &&
				    child->children[0]->Cast<LogicalGet>().function.name == "vindex_index_scan") {
					auto &parent_projection = plan->Cast<LogicalProjection>();
					auto &child_projection = child->Cast<LogicalProjection>();

					column_binding_set_t referenced_bindings;
					for (auto &expr : parent_projection.expressions) {
						ExpressionIterator::EnumerateExpression(expr, [&](Expression &expr_ref) {
							if (expr_ref.type == ExpressionType::BOUND_COLUMN_REF) {
								auto &bound_column_ref = expr_ref.Cast<BoundColumnRefExpression>();
								referenced_bindings.insert(bound_column_ref.binding);
							}
						});
					}

					auto child_bindings = child_projection.GetColumnBindings();
					for (idx_t i = 0; i < child_projection.expressions.size(); i++) {
						auto &expr = child_projection.expressions[i];
						auto &outgoing_binding = child_bindings[i];
						if (referenced_bindings.find(outgoing_binding) == referenced_bindings.end()) {
							expr = make_uniq_base<Expression, BoundConstantExpression>(Value(LogicalType::TINYINT));
						}
					}
					return;
				}
			}
		}
		for (auto &child : plan->children) {
			MergeProjections(child);
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		auto did_use_scan = OptimizeChildren(input.context, plan);
		if (did_use_scan) {
			MergeProjections(plan);
		}
	}
};

void RegisterScanOptimizer(DatabaseInstance &db) {
	ExtensionCallbackManager::Get(db).Register(VectorIndexScanOptimizer());
}

} // namespace vindex
} // namespace duckdb
