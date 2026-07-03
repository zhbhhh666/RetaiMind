import streamlit as st
from datetime import datetime, timedelta
from core.data_loader import DataLoader
from core.report_generator import DailyReportGenerator
from core.config import Config

st.set_page_config(page_title="运营日报生成", page_icon="📊", layout="wide")

st.title("📊 运营日报生成")
st.subheader("AIGC 自动生成运营日报与策略建议")

api_key = Config.DEEPSEEK_API_KEY

if not api_key:
    st.error("请先配置 DeepSeek API Key！")
    st.stop()

data_loader = DataLoader()
report_generator = DailyReportGenerator(api_key)

default_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
date_str = st.date_input("选择日期", value=datetime.now() - timedelta(days=1))
date_str = date_str.strftime("%Y-%m-%d")

with st.spinner("加载运营数据..."):
    data = data_loader.load_daily_operations(date_str)

st.subheader("昨日核心指标")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("销售额", f"¥{data['gmv']:,.0f}", f"{data['gmv_mom']:+.1f}%")
    st.metric("订单量", f"{data['orders']:,}", f"{data['orders_mom']:+.1f}%")

with col2:
    st.metric("客单价", f"¥{data['aov']:.1f}", f"{data['aov_mom']:+.1f}%")
    st.metric("新增会员", f"{data['new_members']}", f"{data['new_members_mom']:+.1f}%")

with col3:
    st.metric("会员复购率", f"{data['repurchase_rate']:.1f}%", f"{data['repurchase_mom']:+.1f}%")

st.subheader("品类销售 TOP3")
st.text(data['top_categories'])

st.subheader("异常事件")
st.text(data['abnormal_events'])

if st.button("生成运营日报"):
    with st.spinner("AI 正在撰写运营日报..."):
        try:
            report = report_generator.generate(data)
            
            st.subheader("📝 运营日报")
            st.markdown(report)
            
            st.download_button(
                label="下载日报",
                data=report,
                file_name=f"运营日报_{date_str}.md",
                mime="text/markdown"
            )
        
        except Exception as e:
            st.error(f"日报生成失败：{str(e)}")

st.subheader("日报预览")
st.info("点击上方「生成运营日报」按钮，AI 将基于上述数据生成完整的运营日报，包括：")
st.write("1. 昨日概览 - 一句话总结经营状况")
st.write("2. 核心指标解读 - 分析关键指标变化原因")
st.write("3. 品类洞察 - TOP3 品类表现分析")
st.write("4. 风险提示 - 基于异常事件的风险预警")
st.write("5. 今日策略建议 - 3-5 条可执行建议")