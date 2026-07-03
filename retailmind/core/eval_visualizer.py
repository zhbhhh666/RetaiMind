"""评估结果可视化仪表盘。

使用 Plotly 生成交互式 HTML 仪表盘，包含：
- 评分卡片 + 综合得分仪表盘
- RAG / Agent 雷达图（独立维度，不混淆）
- 指标柱状图（同单位同图，不混用 % 和 ms）
- 品类 / 任务类型分组对比
- 工具通过率详情
- 完整 HTML 页面（自定义 CSS，响应式布局）
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional, List

from core.logger import get_logger

logger = get_logger(__name__)

# ---- 统一配色 ----
COLORS = {
    "rag_primary": "#6366f1",
    "rag_light": "#818cf8",
    "rag_bg": "rgba(99,102,241,0.12)",
    "agent_primary": "#10b981",
    "agent_light": "#34d399",
    "agent_bg": "rgba(16,185,129,0.12)",
    "amber": "#f59e0b",
    "amber_light": "#fbbf24",
    "red": "#ef4444",
    "gray": "#94a3b8",
    "bg": "#f8fafc",
    "card_bg": "#ffffff",
    "text": "#1e293b",
    "text_light": "#64748b",
    "border": "#e2e8f0",
}


class EvalVisualizer:
    """评估结果可视化器。

    将 RAG 和 Agent 评估结果渲染为交互式 HTML 仪表盘。
    """

    def __init__(self, output_dir: str = "./data/eval/results"):
        """初始化可视化器。

        Args:
            output_dir: HTML 文件输出目录。

        Raises:
            ValueError: 当 output_dir 为空时。
        """
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir 不能为空")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("可视化器初始化，输出目录: %s", output_dir)

    def render_dashboard(self,
                         rag_result: Optional[Dict] = None,
                         agent_result: Optional[Dict] = None,
                         filename: str = None) -> str:
        """生成交互式 HTML 评估仪表盘。

        Args:
            rag_result: RAG 评估结果（evaluate_all() 的返回值）。
            agent_result: Agent 评估结果（evaluate_all() 的返回值）。
            filename: 输出文件名（不含扩展名），默认使用时间戳。

        Returns:
            生成的 HTML 文件路径。

        Raises:
            ValueError: 当两种结果都为空时。
        """
        if not rag_result and not agent_result:
            raise ValueError("rag_result 和 agent_result 不能同时为空")

        if filename is None:
            filename = f"eval_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        import plotly.graph_objects as go
        import plotly.io as pio

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        has_rag = rag_result is not None
        has_agent = agent_result is not None

        # ---- 生成各图表的 HTML div ----
        charts_html = []

        # 1. 评分卡片
        charts_html.append(self._build_score_cards(rag_result, agent_result))

        # 2. 仪表盘 + 雷达图
        gauge_row = '<div class="chart-row">'
        if has_rag:
            gauge_row += self._build_gauge_div(
                rag_result.get("overall", {}).get("overall_score", 0),
                "RAG 系统", COLORS["rag_primary"],
            )
        if has_agent:
            gauge_row += self._build_gauge_div(
                agent_result.get("overall", {}).get("overall_score", 0),
                "库存 Agent", COLORS["agent_primary"],
            )
        gauge_row += '</div>'
        charts_html.append(gauge_row)

        # 3. 雷达图
        radar_row = '<div class="chart-row">'
        if has_rag:
            radar_row += self._wrap_chart_div(
                self._build_rag_radar(rag_result), "RAG 多维度雷达"
            )
        if has_agent:
            radar_row += self._wrap_chart_div(
                self._build_agent_radar(agent_result), "Agent 多维度雷达"
            )
        radar_row += '</div>'
        charts_html.append(radar_row)

        # 4. RAG 指标柱状图
        if has_rag:
            rag_row = '<div class="chart-row">'
            rag_row += self._wrap_chart_div(
                self._build_rag_retrieval_bars(rag_result), "RAG 检索指标"
            )
            rag_row += self._wrap_chart_div(
                self._build_rag_generation_bars(rag_result), "RAG 生成指标"
            )
            rag_row += '</div>'
            charts_html.append(rag_row)

            # 品类得分
            cat_chart = self._build_rag_category_bars(rag_result)
            if cat_chart:
                charts_html.append(
                    self._wrap_chart_fullwidth(cat_chart, "RAG 按品类得分")
                )

        # 5. Agent 指标柱状图
        if has_agent:
            agent_row = '<div class="chart-row">'
            agent_row += self._wrap_chart_div(
                self._build_agent_selection_bars(agent_result), "Agent 工具选择指标"
            )
            agent_row += self._wrap_chart_div(
                self._build_agent_function_bars(agent_result), "Agent 工具函数通过率"
            )
            agent_row += '</div>'
            charts_html.append(agent_row)

            # 任务类型
            tt_chart = self._build_agent_tasktype_bars(agent_result)
            if tt_chart:
                charts_html.append(
                    self._wrap_chart_fullwidth(tt_chart, "Agent 按任务类型 F1")
                )

        # ---- 组装完整 HTML ----
        html = self._build_html_page(timestamp, "\n".join(charts_html))

        html_path = os.path.join(self.output_dir, f"{filename}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("评估仪表盘已保存: %s", html_path)
        return html_path

    # ========== HTML 页面骨架 ==========

    def _build_html_page(self, timestamp: str, charts_html: str) -> str:
        """构建完整 HTML 页面。"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RetaiMind 模型评估仪表盘</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
    background: {COLORS["bg"]};
    color: {COLORS["text"]};
    padding: 24px;
  }}
  .header {{
    text-align: center;
    margin-bottom: 28px;
    padding: 28px 0;
    background: linear-gradient(135deg, #6366f1 0%, #10b981 100%);
    border-radius: 16px;
    color: white;
    box-shadow: 0 4px 20px rgba(99,102,241,0.18);
  }}
  .header h1 {{ font-size: 26px; font-weight: 700; margin-bottom: 6px; }}
  .header .meta {{ font-size: 14px; opacity: 0.85; }}

  .score-cards {{
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 24px;
    flex-wrap: wrap;
  }}
  .score-card {{
    background: {COLORS["card_bg"]};
    border-radius: 14px;
    padding: 20px 32px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center;
    min-width: 200px;
    border-top: 4px solid {COLORS["gray"]};
  }}
  .score-card.rag {{ border-top-color: {COLORS["rag_primary"]}; }}
  .score-card.agent {{ border-top-color: {COLORS["agent_primary"]}; }}
  .score-card .label {{ font-size: 13px; color: {COLORS["text_light"]}; margin-bottom: 6px; }}
  .score-card .value {{ font-size: 32px; font-weight: 800; }}
  .score-card .unit {{ font-size: 14px; color: {COLORS["text_light"]}; font-weight: 400; }}
  .score-card.rag .value {{ color: {COLORS["rag_primary"]}; }}
  .score-card.agent .value {{ color: {COLORS["agent_primary"]}; }}

  .chart-row {{
    display: flex;
    gap: 20px;
    margin-bottom: 20px;
    flex-wrap: wrap;
    align-items: stretch;
  }}
  .chart-box {{
    background: {COLORS["card_bg"]};
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    flex: 1;
    min-width: 420px;
  }}
  .chart-box.full {{ flex: 1 1 100%; }}
  .chart-title {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 12px;
    padding-left: 10px;
    border-left: 3px solid {COLORS["rag_primary"]};
  }}
  .chart-title.agent {{ border-left-color: {COLORS["agent_primary"]}; }}
  .chart-content {{ width: 100%; }}

  .footer {{
    text-align: center;
    padding: 24px;
    color: {COLORS["text_light"]};
    font-size: 13px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>RetaiMind 模型评估仪表盘</h1>
  <div class="meta">生成时间: {timestamp} &nbsp;|&nbsp; 由 RetaiMind 评估体系自动生成</div>
</div>

{charts_html}

<div class="footer">
  RetaiMind Evaluation System &mdash; 交互式仪表盘支持缩放、悬停查看数值
</div>

</body>
</html>"""

    # ========== 评分卡片 ==========

    def _build_score_cards(self, rag_result, agent_result) -> str:
        """构建顶部评分卡片。"""
        cards = '<div class="score-cards">'
        if rag_result:
            score = rag_result.get("overall", {}).get("overall_score", 0)
            cards += f'<div class="score-card rag"><div class="label">RAG 系统综合得分</div><div class="value">{score}<span class="unit"> /100</span></div></div>'
        if agent_result:
            score = agent_result.get("overall", {}).get("overall_score", 0)
            cards += f'<div class="score-card agent"><div class="label">库存 Agent 综合得分</div><div class="value">{score}<span class="unit"> /100</span></div></div>'
        cards += '</div>'
        return cards

    # ========== 仪表盘（Gauge） ==========

    def _build_gauge_div(self, score: float, title: str, color: str) -> str:
        """构建单个仪表盘 div。"""
        import plotly.graph_objects as go

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title=dict(text=title, font=dict(size=15)),
            number=dict(suffix="/100", font=dict(size=28)),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=COLORS["text_light"]),
                bar=dict(color=color),
                bgcolor="#f1f5f9",
                borderwidth=0,
                steps=[
                    dict(range=[0, 60], color="#fee2e2"),
                    dict(range=[60, 80], color="#fef3c7"),
                    dict(range=[80, 100], color="#d1fae5"),
                ],
                threshold=dict(line=dict(color=COLORS["red"], width=3), value=80),
            ),
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=20))
        div_id = f"gauge_{title.replace(' ', '_')}"
        return self._fig_to_div(fig, div_id)

    # ========== 雷达图 ==========

    def _build_rag_radar(self, rag_result) -> str:
        """构建 RAG 雷达图。"""
        import plotly.graph_objects as go

        overall = rag_result.get("overall", {})
        categories = ["检索命中率", "MRR", "关键词命中", "可回答率"]
        values = [
            overall.get("retrieval_hit_rate", 0) * 100,
            overall.get("retrieval_mrr", 0) * 100,
            overall.get("generation_keyword_hit_rate", 0) * 100,
            overall.get("generation_answerability_rate", 0) * 100,
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="RAG",
            line=dict(color=COLORS["rag_primary"], width=2),
            fillcolor=COLORS["rag_bg"],
            hovertemplate="<b>%{theta}</b><br>得分: %{r:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], ticksuffix="%", tickfont=dict(size=10)),
                angularaxis=dict(rotation=90, direction="clockwise", tickfont=dict(size=11)),
            ),
            height=340,
            margin=dict(l=40, r=40, t=20, b=20),
            showlegend=False,
        )
        return self._fig_to_div(fig, "radar_rag")

    def _build_agent_radar(self, agent_result) -> str:
        """构建 Agent 雷达图。"""
        import plotly.graph_objects as go

        overall = agent_result.get("overall", {})
        categories = ["工具匹配率", "精确率", "召回率", "F1", "函数通过率"]
        values = [
            overall.get("tool_selection_exact_match_rate", 0) * 100,
            agent_result.get("tool_selection", {}).get("avg_precision", 0) * 100,
            agent_result.get("tool_selection", {}).get("avg_recall", 0) * 100,
            overall.get("tool_selection_f1", 0) * 100,
            overall.get("tool_function_pass_rate", 0) * 100,
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Agent",
            line=dict(color=COLORS["agent_primary"], width=2),
            fillcolor=COLORS["agent_bg"],
            hovertemplate="<b>%{theta}</b><br>得分: %{r:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 100], ticksuffix="%", tickfont=dict(size=10)),
                angularaxis=dict(rotation=90, direction="clockwise", tickfont=dict(size=11)),
            ),
            height=340,
            margin=dict(l=40, r=40, t=20, b=20),
            showlegend=False,
        )
        return self._fig_to_div(fig, "radar_agent")

    # ========== RAG 柱状图 ==========

    def _build_rag_retrieval_bars(self, rag_result) -> str:
        """构建 RAG 检索指标柱状图（仅百分率指标）。"""
        import plotly.graph_objects as go

        retrieval = rag_result.get("retrieval", {})
        metrics = ["Hit Rate", "MRR"]
        values = [
            retrieval.get("hit_rate", 0) * 100,
            retrieval.get("mrr", 0) * 100,
        ]
        colors = [COLORS["rag_primary"], COLORS["rag_light"]]

        fig = go.Figure(go.Bar(
            x=metrics, y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=13, color=COLORS["text"]),
            hovertemplate="<b>%{x}</b><br>得分: %{y:.1f}%<extra></extra>",
            width=0.5,
        ))
        fig.update_layout(
            yaxis=dict(title="得分(%)", range=[0, 110], gridcolor=COLORS["border"]),
            xaxis=dict(tickfont=dict(size=12)),
            height=320,
            margin=dict(l=40, r=20, t=10, b=30),
            showlegend=False,
            template="plotly_white",
        )
        # 添加检索耗时注释
        avg_time = retrieval.get("avg_retrieval_time_ms", 0)
        fig.add_annotation(
            text=f"平均检索耗时: {avg_time:.1f} ms",
            xref="paper", yref="paper", x=0.99, y=0.02,
            xanchor="right", yanchor="bottom",
            font=dict(size=11, color=COLORS["text_light"]),
            bgcolor="rgba(241,245,249,0.8)",
            bordercolor=COLORS["border"], borderwidth=1, borderpad=4,
        )
        return self._fig_to_div(fig, "rag_retrieval")

    def _build_rag_generation_bars(self, rag_result) -> str:
        """构建 RAG 生成指标柱状图（仅百分率指标）。"""
        import plotly.graph_objects as go

        generation = rag_result.get("generation", {})
        metrics = ["关键词命中率", "可回答率"]
        values = [
            generation.get("keyword_hit_rate", 0) * 100,
            generation.get("answerability_rate", 0) * 100,
        ]
        colors = [COLORS["amber"], COLORS["amber_light"]]

        fig = go.Figure(go.Bar(
            x=metrics, y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=13, color=COLORS["text"]),
            hovertemplate="<b>%{x}</b><br>得分: %{y:.1f}%<extra></extra>",
            width=0.5,
        ))
        fig.update_layout(
            yaxis=dict(title="得分(%)", range=[0, 110], gridcolor=COLORS["border"]),
            xaxis=dict(tickfont=dict(size=12)),
            height=320,
            margin=dict(l=40, r=20, t=10, b=30),
            showlegend=False,
            template="plotly_white",
        )
        avg_len = generation.get("avg_answer_length", 0)
        avg_qtime = generation.get("avg_query_time_ms", 0)
        fig.add_annotation(
            text=f"平均回答: {avg_len:.0f} 字 | 平均查询: {avg_qtime:.0f} ms",
            xref="paper", yref="paper", x=0.99, y=0.02,
            xanchor="right", yanchor="bottom",
            font=dict(size=11, color=COLORS["text_light"]),
            bgcolor="rgba(241,245,249,0.8)",
            bordercolor=COLORS["border"], borderwidth=1, borderpad=4,
        )
        return self._fig_to_div(fig, "rag_generation")

    # ========== Agent 柱状图 ==========

    def _build_agent_selection_bars(self, agent_result) -> str:
        """构建 Agent 工具选择指标柱状图。"""
        import plotly.graph_objects as go

        selection = agent_result.get("tool_selection", {})
        metrics = ["精确匹配率", "精确率", "召回率", "F1"]
        values = [
            selection.get("exact_match_rate", 0) * 100,
            selection.get("avg_precision", 0) * 100,
            selection.get("avg_recall", 0) * 100,
            selection.get("avg_f1", 0) * 100,
        ]
        colors = [COLORS["agent_primary"], COLORS["agent_light"], "#6ee7b7", "#a7f3d0"]

        fig = go.Figure(go.Bar(
            x=metrics, y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=13, color=COLORS["text"]),
            hovertemplate="<b>%{x}</b><br>得分: %{y:.1f}%<extra></extra>",
            width=0.5,
        ))
        fig.update_layout(
            yaxis=dict(title="得分(%)", range=[0, 110], gridcolor=COLORS["border"]),
            xaxis=dict(tickfont=dict(size=12)),
            height=320,
            margin=dict(l=40, r=20, t=10, b=30),
            showlegend=False,
            template="plotly_white",
        )
        total = selection.get("total_tasks", 0)
        fig.add_annotation(
            text=f"任务总数: {total}",
            xref="paper", yref="paper", x=0.99, y=0.02,
            xanchor="right", yanchor="bottom",
            font=dict(size=11, color=COLORS["text_light"]),
            bgcolor="rgba(241,245,249,0.8)",
            bordercolor=COLORS["border"], borderwidth=1, borderpad=4,
        )
        return self._fig_to_div(fig, "agent_selection")

    def _build_agent_function_bars(self, agent_result) -> str:
        """构建 Agent 工具函数通过率柱状图。"""
        import plotly.graph_objects as go

        func = agent_result.get("tool_functions", {})
        per_tool = func.get("per_tool", {})

        fig = go.Figure()
        if per_tool:
            tool_names = list(per_tool.keys())
            passed = [1 if per_tool[t].get("passed") else 0 for t in tool_names]
            colors = [COLORS["agent_primary"] if p else COLORS["red"] for p in passed]

            fig.add_trace(go.Bar(
                x=tool_names,
                y=[p * 100 for p in passed],
                marker_color=colors,
                text=["✅ 通过" if p else "❌ 失败" for p in passed],
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
                width=0.6,
                showlegend=False,
            ))
            fig.update_layout(
                yaxis=dict(title="通过率(%)", range=[0, 115], gridcolor=COLORS["border"]),
                xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
                height=340,
                margin=dict(l=40, r=20, t=10, b=80),
                template="plotly_white",
            )
        else:
            pass_rate = func.get("pass_rate", 0) * 100
            fig.add_trace(go.Bar(
                x=["通过率"], y=[pass_rate],
                marker_color=COLORS["agent_primary"],
                text=[f"{pass_rate:.1f}%"],
                textposition="outside",
                width=0.4,
                showlegend=False,
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 110]),
                height=340,
                margin=dict(l=40, r=20, t=10, b=30),
                template="plotly_white",
            )

        total = func.get("total_tools", 0)
        passed_n = func.get("passed_tools", 0)
        fig.add_annotation(
            text=f"通过: {passed_n}/{total}",
            xref="paper", yref="paper", x=0.99, y=0.02,
            xanchor="right", yanchor="bottom",
            font=dict(size=11, color=COLORS["text_light"]),
            bgcolor="rgba(241,245,249,0.8)",
            bordercolor=COLORS["border"], borderwidth=1, borderpad=4,
        )
        return self._fig_to_div(fig, "agent_functions")

    # ========== 分组对比 ==========

    def _build_rag_category_bars(self, rag_result) -> str:
        """构建 RAG 品类得分分组柱状图。"""
        import plotly.graph_objects as go

        retrieval = rag_result.get("retrieval", {})
        gen = rag_result.get("generation", {})
        ret_per_q = {item["id"]: item for item in retrieval.get("per_question", [])}
        gen_per_q = {item["id"]: item for item in gen.get("per_question", [])}

        categories_data = self._load_rag_categories()

        if not categories_data:
            return ""

        cat_names = list(categories_data.keys())
        hit_rates = []
        keyword_rates = []
        for cat, q_ids in categories_data.items():
            hits = sum(1 for qid in q_ids if ret_per_q.get(qid, {}).get("hit", False))
            kws = sum(1 for qid in q_ids if gen_per_q.get(qid, {}).get("keyword_score", 0) > 0)
            hit_rates.append(hits / len(q_ids) * 100 if q_ids else 0)
            keyword_rates.append(kws / len(q_ids) * 100 if q_ids else 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat_names, y=hit_rates,
            name="检索命中率",
            marker_color=COLORS["rag_primary"],
            text=[f"{v:.0f}%" for v in hit_rates],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>命中率: %{y:.1f}%<extra></extra>",
            offsetgroup=0,
        ))
        fig.add_trace(go.Bar(
            x=cat_names, y=keyword_rates,
            name="关键词命中率",
            marker_color=COLORS["amber"],
            text=[f"{v:.0f}%" for v in keyword_rates],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>关键词命中: %{y:.1f}%<extra></extra>",
            offsetgroup=1,
        ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(title="得分(%)", range=[0, 115], gridcolor=COLORS["border"]),
            xaxis=dict(tickfont=dict(size=12)),
            height=340,
            margin=dict(l=40, r=20, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        return self._fig_to_div(fig, "rag_category")

    def _build_agent_tasktype_bars(self, agent_result) -> str:
        """构建 Agent 任务类型 F1 分组柱状图。"""
        import plotly.graph_objects as go

        selection = agent_result.get("tool_selection", {})
        per_task = selection.get("per_task", [])

        task_types = self._load_agent_task_types()

        if not task_types:
            return ""

        type_names = list(task_types.keys())
        f1_values = []
        match_values = []
        for ttype, task_ids in task_types.items():
            f1s = [
                next((t["f1"] for t in per_task if t["id"] == tid), 0)
                for tid in task_ids
            ]
            matches = [
                1 if next((t.get("exact_match", False) for t in per_task if t["id"] == tid), False) else 0
                for tid in task_ids
            ]
            f1_values.append(sum(f1s) / len(f1s) * 100 if f1s else 0)
            match_values.append(sum(matches) / len(matches) * 100 if matches else 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=type_names, y=f1_values,
            name="平均 F1",
            marker_color=COLORS["agent_primary"],
            text=[f"{v:.1f}%" for v in f1_values],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>F1: %{y:.1f}%<extra></extra>",
            offsetgroup=0,
        ))
        fig.add_trace(go.Bar(
            x=type_names, y=match_values,
            name="精确匹配率",
            marker_color=COLORS["agent_light"],
            text=[f"{v:.1f}%" for v in match_values],
            textposition="outside",
            textfont=dict(size=12),
            hovertemplate="<b>%{x}</b><br>匹配率: %{y:.1f}%<extra></extra>",
            offsetgroup=1,
        ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(title="得分(%)", range=[0, 115], gridcolor=COLORS["border"]),
            xaxis=dict(tickfont=dict(size=12)),
            height=340,
            margin=dict(l=40, r=20, t=10, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
        )
        return self._fig_to_div(fig, "agent_tasktype")

    # ========== 辅助方法 ==========

    def _fig_to_div(self, fig, div_id: str) -> str:
        """将 Figure 转为 HTML div 字符串。"""
        import plotly.io as pio
        return pio.to_html(
            fig,
            include_plotlyjs=False,
            full_html=False,
            div_id=div_id,
            config=dict(displayModeBar=True, responsive=True),
        )

    def _wrap_chart_div(self, chart_html: str, title: str, is_agent: bool = False) -> str:
        """将图表 HTML 包装在卡片容器中。"""
        css_class = "agent" if is_agent else ""
        return f"""<div class="chart-box">
  <div class="chart-title {css_class}">{title}</div>
  <div class="chart-content">{chart_html}</div>
</div>"""

    def _wrap_chart_fullwidth(self, chart_html: str, title: str, is_agent: bool = False) -> str:
        """将图表包装为全宽卡片。"""
        css_class = "agent" if is_agent else ""
        return f"""<div class="chart-row">
  <div class="chart-box full">
    <div class="chart-title {css_class}">{title}</div>
    <div class="chart-content">{chart_html}</div>
  </div>
</div>"""

    def _load_rag_categories(self) -> Dict:
        """从评估数据集加载 RAG 品类信息。"""
        dataset_path = os.path.join(
            os.path.dirname(self.output_dir),
            "rag_eval_dataset.json"
        )
        if not os.path.exists(dataset_path):
            return {}
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        categories_data = {}
        for q in dataset.get("questions", []):
            cat = q.get("category", "未分类")
            if cat not in categories_data:
                categories_data[cat] = []
            categories_data[cat].append(q["id"])
        return categories_data

    def _load_agent_task_types(self) -> Dict:
        """从评估数据集加载 Agent 任务类型信息。"""
        dataset_path = os.path.join(
            os.path.dirname(self.output_dir),
            "agent_eval_dataset.json"
        )
        if not os.path.exists(dataset_path):
            return {}
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        task_types = {}
        for t in dataset.get("tasks", []):
            ttype = t.get("task_type", "未分类")
            if ttype not in task_types:
                task_types[ttype] = []
            task_types[ttype].append(t["id"])
        return task_types
