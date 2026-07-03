import streamlit as st
import pandas as pd
import plotly.express as px
from core.data_loader import DataLoader
from core.member_analyzer import MemberAnalyzer
from core.config import Config

st.set_page_config(page_title="会员数据分析", page_icon="👥", layout="wide")

st.title("👥 会员数据分析")
st.subheader("会员 RFM 分析与分群洞察")

data_loader = DataLoader()
analyzer = MemberAnalyzer()

with st.spinner("加载会员数据..."):
    transactions_df = data_loader.load_member_transactions()
    
st.success(f"加载成功！共 {len(transactions_df)} 条交易记录，{transactions_df['customer_id'].nunique()} 位会员")

if st.button("开始分析"):
    with st.spinner("正在进行 RFM 分析..."):
        result = analyzer.analyze(transactions_df)
        rfm_df = result["rfm_data"]
        segment_stats = result["segment_stats"]
        churn_risk = result["churn_risk"]
        distribution = result["distribution"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("会员分群分布")
        fig = px.pie(
            values=distribution[1],
            names=distribution[0],
            title="会员分群占比",
            color_discrete_sequence=px.colors.qualitative.D3
        )
        fig.update_layout(width=400, height=400)
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("核心指标概览")
        total_stats = segment_stats["total"]
        metrics = {
            "总会员数": total_stats["members"],
            "平均最近购买天数": f"{total_stats['avg_recency']} 天",
            "平均购买频次": f"{total_stats['avg_frequency']} 次",
            "平均消费金额": f"¥{total_stats['avg_monetary']:.0f}"
        }
        for metric, value in metrics.items():
            st.metric(label=metric, value=value)
    
    st.subheader("分群详细数据")
    segments_df = pd.DataFrame([{
        "分群名称": name,
        "会员数量": stats["count"],
        "占比": f"{stats['percentage']}%",
        "平均最近购买": f"{stats['avg_recency']}天",
        "平均购买频次": f"{stats['avg_frequency']}次",
        "平均消费金额": f"¥{stats['avg_monetary']:.0f}"
    } for name, stats in segment_stats.items() if name != "total"])
    st.dataframe(segments_df, use_container_width=True)
    
    st.subheader("流失风险会员")
    if not churn_risk.empty:
        churn_df = churn_risk.head(20)[["recency", "frequency", "monetary", "rfm_total"]].reset_index()
        churn_df.columns = ["会员ID", "最近购买天数", "购买频次", "消费金额", "RFM总分"]
        st.dataframe(churn_df, use_container_width=True)
        st.warning(f"共有 {len(churn_risk)} 位会员存在流失风险（超过90天未购买）")
    else:
        st.success("暂无流失风险会员")
    
    st.subheader("RFM 散点图")
    fig_scatter = px.scatter(
        rfm_df,
        x="recency",
        y="monetary",
        color="segment_name",
        size="frequency",
        hover_data=["frequency"],
        title="会员 RFM 分布",
        labels={"recency": "最近购买天数", "monetary": "消费金额", "frequency": "购买频次"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)