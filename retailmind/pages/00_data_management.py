import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from core.data_loader import DataLoader

st.set_page_config(page_title="数据管理中心", page_icon="📁", layout="wide")

st.title("📁 数据管理中心")
st.subheader("上传、管理和重置业务数据")

data_loader = DataLoader()

tab1, tab2, tab3, tab4 = st.tabs(["👥 会员交易数据", "📦 库存数据", "📈 销售预测数据", "📊 运营数据"])


def render_data_info(summary, sample_df):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("数据来源", summary["source"])
    with col2:
        st.metric("数据行数", f"{summary['row_count']:,}")
    with col3:
        st.metric("字段数量", len(summary["columns"]))

    if summary["extra"]:
        st.markdown("**数据概览**")
        extra_cols = st.columns(len(summary["extra"]))
        for i, (k, v) in enumerate(summary["extra"].items()):
            with extra_cols[i]:
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    st.metric(label, f"{v:,.2f}")
                elif isinstance(v, int):
                    st.metric(label, f"{v:,}")
                else:
                    st.metric(label, str(v))

    with st.expander("查看数据预览（前10行）", expanded=False):
        st.dataframe(sample_df.head(10), use_container_width=True)


with tab1:
    st.header("👥 会员交易数据")
    st.caption("支持字段：customer_id, last_purchase_date, frequency, monetary, preferred_category")

    member_summary = data_loader.get_data_summary("member_transactions")
    member_df = data_loader.load_member_transactions()
    render_data_info(member_summary, member_df)

    st.markdown("---")
    st.subheader("上传新数据")
    uploaded_member = st.file_uploader("选择 CSV 文件", type=["csv"], key="member_upload",
                                       help="必填字段：customer_id, last_purchase_date, frequency, monetary")

    if uploaded_member is not None:
        try:
            df_upload = pd.read_csv(uploaded_member)
            valid, errors = data_loader.validate_data("member_transactions", df_upload)
            if valid:
                st.success(f"✓ 校验通过！共 {len(df_upload)} 条记录")
                st.dataframe(df_upload.head(5), use_container_width=True)
                if st.button("确认上传会员数据", type="primary", key="member_confirm"):
                    filepath = data_loader.upload_data("member_transactions", df_upload)
                    st.success(f"会员数据已成功保存到：`{filepath}`")
                    st.rerun()
            else:
                st.error("数据校验失败：")
                for err in errors:
                    st.write(f"- {err}")
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")

    st.markdown("---")
    if st.button("🔄 重置为示例数据", key="member_reset"):
        data_loader.reset_data("member_transactions")
        st.success("会员数据已重置为示例数据")
        st.rerun()


with tab2:
    st.header("📦 库存数据")
    st.caption("支持字段：sku_id, stock_qty, safety_stock, weekly_sales, category")

    inv_summary = data_loader.get_data_summary("inventory")
    inv_df = data_loader.load_inventory_data()
    render_data_info(inv_summary, inv_df)

    st.markdown("---")
    st.subheader("上传新数据")
    uploaded_inv = st.file_uploader("选择 CSV 文件", type=["csv"], key="inventory_upload",
                                     help="必填字段：sku_id, stock_qty, safety_stock, weekly_sales, category")

    if uploaded_inv is not None:
        try:
            df_upload = pd.read_csv(uploaded_inv)
            valid, errors = data_loader.validate_data("inventory", df_upload)
            if valid:
                st.success(f"✓ 校验通过！共 {len(df_upload)} 条记录")
                st.dataframe(df_upload.head(5), use_container_width=True)
                if st.button("确认上传库存数据", type="primary", key="inventory_confirm"):
                    filepath = data_loader.upload_data("inventory", df_upload)
                    st.success(f"库存数据已成功保存到：`{filepath}`")
                    st.rerun()
            else:
                st.error("数据校验失败：")
                for err in errors:
                    st.write(f"- {err}")
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")

    st.markdown("---")
    if st.button("🔄 重置为示例数据", key="inventory_reset"):
        data_loader.reset_data("inventory")
        st.success("库存数据已重置为示例数据")
        st.rerun()


with tab3:
    st.header("📈 销售预测数据")
    st.caption("支持字段：sku_id, forecast_7d")

    fc_summary = data_loader.get_data_summary("sales_forecast")
    fc_df = data_loader.load_sales_forecast()
    render_data_info(fc_summary, fc_df)

    st.markdown("---")
    st.subheader("上传新数据")
    uploaded_fc = st.file_uploader("选择 CSV 文件", type=["csv"], key="forecast_upload",
                                    help="必填字段：sku_id, forecast_7d")

    if uploaded_fc is not None:
        try:
            df_upload = pd.read_csv(uploaded_fc)
            valid, errors = data_loader.validate_data("sales_forecast", df_upload)
            if valid:
                st.success(f"✓ 校验通过！共 {len(df_upload)} 条记录")
                st.dataframe(df_upload.head(5), use_container_width=True)
                if st.button("确认上传预测数据", type="primary", key="forecast_confirm"):
                    filepath = data_loader.upload_data("sales_forecast", df_upload)
                    st.success(f"销售预测数据已成功保存到：`{filepath}`")
                    st.rerun()
            else:
                st.error("数据校验失败：")
                for err in errors:
                    st.write(f"- {err}")
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")

    st.markdown("---")
    if st.button("🔄 重置为示例数据", key="forecast_reset"):
        data_loader.reset_data("sales_forecast")
        st.success("销售预测数据已重置为示例数据")
        st.rerun()


with tab4:
    st.header("📊 运营数据")
    st.caption("支持字段：gmv, orders, aov, new_members, repurchase_rate 等")

    default_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    op_date = st.date_input("选择日期", value=datetime.now() - timedelta(days=1), key="op_date")
    op_date_str = op_date.strftime("%Y-%m-%d")

    op_summary = data_loader.get_data_summary("operations", op_date_str)
    op_data = data_loader.load_daily_operations(op_date_str)
    op_df = pd.DataFrame([op_data])
    render_data_info(op_summary, op_df)

    st.markdown("---")
    st.subheader("上传新数据")
    uploaded_op = st.file_uploader("选择 CSV 文件", type=["csv"], key="ops_upload",
                                    help="必填字段：gmv, orders, aov, new_members, repurchase_rate")

    if uploaded_op is not None:
        try:
            df_upload = pd.read_csv(uploaded_op)
            valid, errors = data_loader.validate_data("operations", df_upload, op_date_str)
            if valid:
                st.success(f"✓ 校验通过！共 {len(df_upload)} 条记录")
                st.dataframe(df_upload.head(5), use_container_width=True)
                if st.button("确认上传运营数据", type="primary", key="ops_confirm"):
                    filepath = data_loader.upload_data("operations", df_upload, op_date_str)
                    st.success(f"运营数据已成功保存到：`{filepath}`")
                    st.rerun()
            else:
                st.error("数据校验失败：")
                for err in errors:
                    st.write(f"- {err}")
        except Exception as e:
            st.error(f"文件读取失败：{str(e)}")

    st.markdown("---")
    if st.button("🔄 重置为示例数据", key="ops_reset"):
        data_loader.reset_data("operations", op_date_str)
        st.success(f"{op_date_str} 的运营数据已重置为示例数据")
        st.rerun()
