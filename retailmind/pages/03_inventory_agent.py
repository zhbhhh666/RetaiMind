import streamlit as st
import pandas as pd
import plotly.express as px
from core.data_loader import DataLoader
from core.agent_inventory import InventoryAgent
from core.config import Config

st.set_page_config(page_title="库存预警 Agent", page_icon="📦", layout="wide")

st.title("📦 库存预警 Agent")
st.subheader("AI 驱动的库存监控与补货建议")

api_key = Config.DEEPSEEK_API_KEY

if not api_key:
    st.error("请先配置 DeepSeek API Key！")
    st.stop()

data_loader = DataLoader()

with st.spinner("加载库存数据..."):
    inventory_df = data_loader.load_inventory_data()
    forecast_df = data_loader.load_sales_forecast()

st.success(f"加载成功！共 {len(inventory_df)} 个 SKU")

col1, col2 = st.columns(2)

with col1:
    st.subheader("库存状态概览")
    low_stock = inventory_df[inventory_df['stock_qty'] < inventory_df['safety_stock']].shape[0]
    high_stock = inventory_df[inventory_df['stock_qty'] > inventory_df['safety_stock'] * 2].shape[0]
    normal_stock = len(inventory_df) - low_stock - high_stock
    
    st.metric("库存正常", normal_stock)
    st.metric("库存偏低", low_stock)
    st.metric("库存充足", high_stock)

with col2:
    st.subheader("库存分布")
    fig = px.histogram(
        inventory_df,
        x="stock_qty",
        nbins=20,
        title="库存数量分布",
        labels={"stock_qty": "库存数量"}
    )
    fig.update_layout(width=400, height=350)
    st.plotly_chart(fig)

st.subheader("库存预警列表")
warning_df = inventory_df[inventory_df['stock_qty'] < inventory_df['safety_stock']].copy()
warning_df['shortage'] = warning_df['safety_stock'] - warning_df['stock_qty']
warning_df = warning_df[['sku_id', 'category', 'stock_qty', 'safety_stock', 'shortage', 'weekly_sales']]
warning_df.columns = ['SKU ID', '品类', '当前库存', '安全库存', '短缺数量', '周销量']

if not warning_df.empty:
    st.dataframe(warning_df, use_container_width=True)
    st.warning(f"共有 {len(warning_df)} 个 SKU 库存低于安全库存线")
else:
    st.success("🎉 所有 SKU 库存正常！")

st.subheader("AI 库存顾问")
st.write("请向库存顾问提问，例如：")
st.write("- SKU000015 是否需要补货？")
st.write("- 哪些商品需要紧急补货？")
st.write("- 请预测 SKU000020 未来的库存情况")

query = st.text_input("输入您的问题：", placeholder="例如：哪些商品库存不足？")

if query:
    with st.spinner("AI 顾问正在分析..."):
        try:
            agent = InventoryAgent(api_key, inventory_df, forecast_df)
            result = agent.run(query)
            
            st.subheader("AI 顾问回答")
            st.markdown(result["output"])
            
            if "intermediate_steps" in result and result["intermediate_steps"]:
                st.subheader("思考过程")
                for i, step in enumerate(result["intermediate_steps"]):
                    st.write(f"**步骤 {i+1}**: {step}")
        
        except Exception as e:
            st.error(f"AI 顾问回答失败：{str(e)}")