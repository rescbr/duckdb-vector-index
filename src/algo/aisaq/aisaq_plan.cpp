#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/column_index.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "algo/aisaq/aisaq_build.hpp"
#include "algo/aisaq/aisaq_index.hpp"

namespace duckdb {
namespace vindex {
namespace aisaq {

PhysicalOperator &AiSaqIndex::CreatePlan(PlanIndexInput &input) {
	auto &create_index = input.op;
	auto &context = input.context;
	auto &planner = input.planner;

	Value vindex_val, hnsw_val;
	context.TryGetCurrentSetting("vindex_enable_experimental_persistence", vindex_val);
	context.TryGetCurrentSetting("hnsw_enable_experimental_persistence", hnsw_val);
	const bool enable_persistence =
	    (!vindex_val.IsNull() && vindex_val.GetValue<bool>()) || (!hnsw_val.IsNull() && hnsw_val.GetValue<bool>());

	auto is_disk_db = !create_index.table.GetStorage().db.GetStorageManager().InMemory();
	auto is_persistence_disabled = !enable_persistence;

	if (is_disk_db && is_persistence_disabled) {
		throw BinderException("AiSAQ indexes can only be created in in-memory databases, or when the configuration "
		                      "option 'vindex_enable_experimental_persistence' is set to true.");
	}

	for (auto &option : create_index.info->options) {
		auto &k = option.first;
		auto &v = option.second;
		if (StringUtil::CIEquals(k, "metric")) {
			if (v.type() != LogicalType::VARCHAR) {
				throw BinderException("AiSAQ index 'metric' must be a string");
			}
			const auto metric = StringUtil::Lower(v.GetValue<string>());
			if (metric != "l2sq" && metric != "cosine" && metric != "ip") {
				throw BinderException("AiSAQ index 'metric' must be one of: 'l2sq', 'cosine', 'ip'");
			}
		} else if (StringUtil::CIEquals(k, "quantizer")) {
			// Validated by the quantizer factory + AiSaqIndex::InitFromOptions,
			// which rejects flat/rabitq.
		} else if (StringUtil::CIEquals(k, "bits")) {
			if (v.type() != LogicalType::INTEGER) {
				throw BinderException("AiSAQ index 'bits' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "m")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'm' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "eta")) {
			// ScaNN anisotropic loss ratio — full validation in the ScaNN factory.
		} else if (StringUtil::CIEquals(k, "rerank")) {
			if (v.type() != LogicalType::INTEGER) {
				throw BinderException("AiSAQ index 'rerank' must be an integer");
			}
			if (v.GetValue<int32_t>() < 1) {
				throw BinderException("AiSAQ index 'rerank' must be at least 1");
			}
		} else if (StringUtil::CIEquals(k, "label_column")) {
			if (v.type() != LogicalType::VARCHAR) {
				throw BinderException("AiSAQ index 'label_column' must be a string (column name)");
			}
		} else if (StringUtil::CIEquals(k, "aisaq_r")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'aisaq_r' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "aisaq_l")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'aisaq_l' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "aisaq_l_build")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'aisaq_l_build' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "aisaq_alpha")) {
			if (v.type() != LogicalType::FLOAT && v.type() != LogicalType::DOUBLE) {
				// Allow numeric literals; the index constructor range-checks.
			}
		} else if (StringUtil::CIEquals(k, "aisaq_inline_pq")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'aisaq_inline_pq' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "aisaq_beam_width")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'aisaq_beam_width' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "aisaq_io_limit")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'aisaq_io_limit' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "aisaq_entry_points")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'aisaq_entry_points' must be an integer");
			}
		} else if (StringUtil::CIEquals(k, "build_strategy")) {
			if (v.type() != LogicalType::VARCHAR) {
				throw BinderException("AiSAQ index 'build_strategy' must be a string");
			}
			const auto s = StringUtil::Lower(v.GetValue<string>());
			if (s != "auto" && s != "paged" && s != "pq_buffer" && s != "exact_prune") {
				throw BinderException("AiSAQ index 'build_strategy' must be one of: 'auto', 'paged', 'pq_buffer', "
				                      "'exact_prune' (slow, best quality)");
			}
		} else if (StringUtil::CIEquals(k, "ram_budget_mb")) {
			if (v.type() != LogicalType::INTEGER && v.type() != LogicalType::BIGINT) {
				throw BinderException("AiSAQ index 'ram_budget_mb' must be an integer");
			}
		} else {
			throw BinderException("Unknown option for AiSAQ index: '%s'", k);
		}
	}

	if (create_index.expressions.size() != 1) {
		throw BinderException("AiSAQ indexes can only be created over a single column of keys.");
	}
	auto &arr_type = create_index.expressions[0]->return_type;
	if (arr_type.id() != LogicalTypeId::ARRAY) {
		throw BinderException("AiSAQ index keys must be of type FLOAT[N]");
	}
	auto &child_type = ArrayType::GetChildType(arr_type);
	if (child_type.id() != LogicalTypeId::FLOAT) {
		throw BinderException("AiSAQ index key type must be 'FLOAT[N]'");
	}

	// Resolve label_column (if any) to a scan column index.
	idx_t label_scan_idx = DConstants::INVALID_INDEX;
	for (const auto &opt : create_index.info->options) {
		if (StringUtil::CIEquals(opt.first, "label_column") && !opt.second.IsNull()) {
			const auto label_name = opt.second.GetValue<string>();
			if (!label_name.empty()) {
				for (const auto &col : create_index.table.GetColumns().Logical()) {
					if (StringUtil::CIEquals(col.GetName(), label_name)) {
						label_scan_idx = col.Oid();
						if (col.Type() != LogicalType::BIGINT) {
							throw BinderException("AiSAQ label_column '%s' must be BIGINT (got %s)", label_name,
							                      col.Type().ToString());
						}
						break;
					}
				}
				if (label_scan_idx == DConstants::INVALID_INDEX) {
					throw BinderException("AiSAQ label_column '%s' not found in table", label_name);
				}
			}
			break;
		}
	}

	// When label_column is set, inject it into the physical table scan so the
	// projection can reference it. The scan currently outputs [vec, rowid];
	// we insert the label column before rowid → [vec, label, rowid].
	idx_t label_scan_pos = DConstants::INVALID_INDEX;
	if (label_scan_idx != DConstants::INVALID_INDEX) {
		auto &scan = input.table_scan.Cast<PhysicalTableScan>();
		label_scan_pos = scan.column_ids.size() - 1; // position before rowid
		scan.column_ids.insert(scan.column_ids.begin() + label_scan_pos, ColumnIndex(label_scan_idx));
		scan.types.insert(scan.types.begin() + label_scan_pos, LogicalType::BIGINT);
		// Keep scan_types in sync so the rowid BoundReferenceExpression resolves.
		create_index.info->scan_types.insert(create_index.info->scan_types.end() - 1, LogicalType::BIGINT);
	}

	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (auto &expression : create_index.expressions) {
		new_column_types.push_back(expression->return_type);
		select_list.push_back(std::move(expression));
	}
	if (label_scan_pos != DConstants::INVALID_INDEX) {
		new_column_types.push_back(LogicalType::BIGINT);
		select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, label_scan_pos));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(
	    make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, create_index.info->scan_types.size() - 1));

	auto &projection =
	    planner.Make<PhysicalProjection>(new_column_types, std::move(select_list), create_index.estimated_cardinality);
	projection.children.push_back(input.table_scan);

	vector<LogicalType> filter_types;
	vector<unique_ptr<Expression>> filter_select_list;

	for (idx_t i = 0; i < new_column_types.size() - 1; i++) {
		filter_types.push_back(new_column_types[i]);
		auto is_not_null_expr =
		    make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
		auto bound_ref = make_uniq<BoundReferenceExpression>(new_column_types[i], i);
		is_not_null_expr->children.push_back(std::move(bound_ref));
		filter_select_list.push_back(std::move(is_not_null_expr));
	}

	auto &null_filter = planner.Make<PhysicalFilter>(std::move(filter_types), std::move(filter_select_list),
	                                                 create_index.estimated_cardinality);
	null_filter.types.emplace_back(LogicalType::ROW_TYPE);
	null_filter.children.push_back(projection);

	auto &physical_create_index = planner.Make<PhysicalCreateAiSaqIndex>(
	    create_index.types, create_index.table, create_index.info->column_ids, std::move(create_index.info),
	    std::move(create_index.unbound_expressions), create_index.estimated_cardinality);
	physical_create_index.children.push_back(null_filter);
	return physical_create_index;
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
