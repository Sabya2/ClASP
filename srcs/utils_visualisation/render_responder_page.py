

"""
Responder Analysis - Streamlit Page Renderer

This module provides the Streamlit UI for the responder analysis section.
"""


import io
import pandas as pd
import streamlit as st
from srcs.utils_features.responder_analysis import calculate_swc_te, fig_to_png_buffer, check_t1_t2_stability


def render_responder_page(
    df: pd.DataFrame,
    group_col: str | None = None,
    continuous_cols: list | None = None,
    participants_col: str | None = None,
):
    """
    Render the responder analysis page in Streamlit.

    Args:
        df: Full dataset
        group_col: Primary grouping column (e.g., SynAge Young/Old)
        continuous_cols: List of continuous variable names
        participants_col: Column name for participant IDs
    """
    st.header("📊 Responder Analysis")

    with st.expander("How this page works", expanded=False):
        st.markdown(
        """
        This page calculates individual responses to interventions using **Smallest Worthwhile Change (SWC)**
        and **Typical Error (TE)** to classify participants.


        **What you can do here**
        - Select pre/post intervention variables and reliability measures (t1, t2)
        - Choose SWC multiplier (typically 0.2 for small effects, 0.5 for moderate)
        - Set confidence level (50%, 68%, 90%, 95%, 99%) for individual response intervals
        - Choose the **error metric (TE or SEM)** used to calculate confidence intervals
        - Choose whether the response threshold is based on **SWC or zero**
        - Optionally split analysis by groups (e.g., age, treatment arm)
        - Check t1-t2 stability before running analysis (recommended)
        - Download individual results, summary statistics, and responder plots as PNG/CSV


        **How classifications work**

        - 🟢 **Reliable Meaningful Responders**: CI lower bound exceeds the response threshold (true response likely)
        - 🟡 **Potential Responders**: Point estimate exceeds the response threshold but the CI includes non-response (uncertain)
        - 🔴 **Non-responders**: No meaningful change detected (change ≤ threshold or CI entirely below threshold)

        **Response threshold**
        The response threshold can be defined in two ways:
        - **SWC-based**: Uses the Smallest Worthwhile Change as the threshold. This identifies changes that are not only statistically distinguishable from measurement error but also large enough to be considered practically meaningful.
        - **Zero-based**: Uses **0** as the threshold. This assesses whether the observed change is distinguishable from no change, without requiring the change to exceed a predefined SWC.

        **Confidence interval calculation**
        The confidence interval (CI) around an individual's change can be calculated using either **TE (Typical Error)** or **SEM (Standard Error of Measurement)**. The selected confidence level determines the corresponding z-value.

        **1. SEM from test-retest reliability (ICC)**
        The standard psychometric definition is:
        $$\\mathrm{SEM} = \\mathrm{SD}_{\\text{baseline}} \\times \\sqrt{1 - \\mathrm{ICC}}$$
        where:
        - **SD_baseline** is the standard deviation of the baseline measurement (or pooled SD across test-retest measurements)
        - **ICC** is the test-retest reliability estimate, such as ICC(2,1) or ICC(3,1)
        If an ICC is not available and two repeated measurements (t1 and t2) are available, SEM can be estimated from the difference scores:
        $$\\mathrm{SEM} = \\frac{\\mathrm{SD}_{\\text{diff}}}{\\sqrt{2}}$$
        where:
        $$\\mathrm{SD}_{\\text{diff}} = \\mathrm{SD}(t2 - t1)$$

        **2. TE (Typical Error) from test-retest measurements**
        TE can be estimated as the standard deviation of the test-retest change scores:
        $$\\mathrm{TE}_{\\text{change}} = \\mathrm{SD}(t2 - t1)$$
        The individual-change CI is then:
        $$\\mathrm{CI}_{\\text{change}} = \\mathrm{change} \\pm z \\times \\mathrm{TE}_{\\text{change}}$$

        **3. SEM-based CI for individual change**
        When SEM is selected, the CI is calculated as:
        $$\\mathrm{CI}_{\\text{change, SEM}} = \\mathrm{change} \\pm z \\times \\mathrm{SEM}$$
        SEM may therefore be estimated using either:
        $$\\mathrm{SEM} = \\mathrm{SD}_{\\text{baseline}}\\sqrt{1-\\mathrm{ICC}}$$
        or:
        $$\\mathrm{SEM} = \\frac{\\mathrm{SD}_{\\text{diff}}}{\\sqrt{2}}$$

        **Interpretation of the error metrics**
        - **TE**: Represents the observed variability in test-retest change scores.
        - **SEM**: Represents the estimated measurement uncertainty of an individual measurement.
        - **TE-based CI** is therefore directly based on the observed test-retest variability.
        - **SEM-based CI** uses the standard error of measurement and generally produces a narrower interval when SEM = SD_diff / √2.

        **Stability tests (t1 vs t2 reliability check)**
        - **Paired t-test**: Tests whether there is a systematic mean difference between t1 and t2 (should be non-significant, p > 0.05)
        - **Effect size (Cohen's d)**: Quantifies magnitude of any systematic change in standardized units (should be < 0.2 for negligible change)
        - **Correlation (r)**: Assesses whether participants maintain rank order across measurements (r > 0.95 indicates excellent stability)
        - **Coefficient of Variation (CV)**: Expresses variability between paired measurements as % of mean (should be < 5-10% for acceptable stability)
        - **Bland-Altman analysis**: Visualizes agreement between t1 and t2, showing bias and 95% limits of agreement (narrow limits = good agreement)
        - **K-S test**: Tests whether t1 and t2 come from the same distribution (p > 0.05 indicates identical distribution shapes)

        **Required data**
        - Participant ID column
        - Pre-intervention measurements (baseline)
        - Post-intervention measurements (after treatment/exercise)
        - Two reliability measurements (test-retest, no intervention) from a subset to estimate TE/SEM


            """
        )



    if df is None:
        st.warning("Upload data first.")
        return

    # ========================================================================
    # COLUMN SELECTION
    # ========================================================================
    st.subheader("1️⃣ Select Variables")

    default_id_col = participants_col if participants_col else (df.columns[0] if len(df) > 0 else None)

    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        id_col = st.selectbox("Participant ID",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(default_id_col) if default_id_col in df.columns else 0,
            key="resp_id_col",
        )

    with col2:
        pre_col = st.selectbox("Pre-intervention",
            options=continuous_cols if continuous_cols else df.select_dtypes(include="number").columns.tolist(),
            index=0,
            key="resp_pre_col",
        )

    with col3:
        post_col = st.selectbox("Post-intervention",
            options=continuous_cols if continuous_cols else df.select_dtypes(include="number").columns.tolist(),
            index=1 if len(continuous_cols or []) > 1 else 0,
            key="resp_post_col",
        )

    col4, col5, col6 = st.columns(3, gap="small")

    with col4:
        use_relative = st.selectbox("Relative or Raw change",
            options=['Raw baseline', "Relative to baseline"],
            index=0,
            key="use_relative",
            help="Calculate SWC and TE in raw or relative units. " \
                "Raw change: post - pre; Relative change: (post - pre) / pre. " \
                "Raw SWC: swc_multiplier × SD(pre); Relative SWC: swc_multiplier × SD(pre) / mean(pre)",
        )
        use_relative = (use_relative == "Relative to baseline")

    with col5:
        t1_col = st.selectbox("Reliability Test / Experiment 1",
            options=continuous_cols if continuous_cols else df.select_dtypes(include="number").columns.tolist(),
            index=0,
            key="resp_t1_col",
            help="First reliability measurement (test)",
        )

    with col6:
        t2_col = st.selectbox("Reliability ReTest / Experiment 2",
            options=continuous_cols if continuous_cols else df.select_dtypes(include="number").columns.tolist(),
            index=2 if len(continuous_cols or []) > 2 else 0,
            key="resp_t2_col",
            help="Second reliability measurement (retest)",
        )

    st.divider()

    # ========================================================================
    # GROUPING & PARAMETERS
    # ========================================================================
    st.subheader("2️⃣ Analysis Settings")

    g1, g2, g3 = st.columns([1.5, 1.5, 2.5], gap="small")

    with g1:
        higher_is_better = st.checkbox("Higher values indicate improvement",
            value=True,
            key="resp_higher_better",
            help="Uncheck if lower values are better (e.g., body fat %)",
        )

        threshold_type = st.selectbox('Threshold for Responder classification',
            options=['Zero-Based', 'SWC-Based'],
            index = 0, key='responder_threshold_type',
            help='Decide the cutoff for reponder based on domain'
        )

        if threshold_type == 'SWC-Based':
            swc_multiplier = st.slider("SWC multiplier",
                        min_value=0.2,
                        max_value=0.5,
                        value=0.2,
                        step=0.05,
                        key="resp_swc_mult",
                        help="0.2 = small, 0.5 = moderate, 0.8 = large effect threshold",
            )
        else:
            swc_multiplier = 0.5


    with g2:
        error_metric = st.selectbox("Error metric for CI",
                            options=['TE', 'SEM'],
                            index=0,
                            key="ci_error_metric",
                            help=(
                                    "TE (Technical Error) quantifies measurement variability/error, whereas "
                                    "SEM (Standard Error of the Mean) quantifies the uncertainty of the estimated mean. "
                                    "For CIs of means, SEM is typically used."
                                )
        )

        ci_level = st.selectbox("Confidence level",
                    options=[0.5, 0.6827, 0.90, 0.95, 0.99],
                    format_func=lambda x: f"{x:.1%}" if x != 0.6827 else "68.27% (≈1 SD)",
                    index=1,
                    key="resp_ci_level",
                    help="50% CI ≈ change ± 0.67×TE; 68% ≈ change ± TE; 95% ≈ change ± 1.96×TE",
                )



    with g3:
        split_by_group = st.checkbox("Split analysis by group",
                    value=False,
                    key="resp_split_by_group",
                    help="Analyze responder patterns separately for each group",
        )

        # Group selection if splitting
        selected_groups = None
        group_order = None

        if split_by_group:
            gc1, gc2 = st.columns([2, 1.8], gap="small")

            with gc1:
                possible_group_cols = df.columns.tolist()
                default_idx = possible_group_cols.index(group_col) if group_col in possible_group_cols else 0
                selected_group_col = st.selectbox("Group column",
                    options=possible_group_cols,
                    index=default_idx,
                    key="resp_group_col",
                )

            with gc2:
                available_groups = df[selected_group_col].dropna().unique().tolist()
                default_groups = available_groups[:3]
                selected_groups = st.multiselect("Groups to plot",
                    options=available_groups,
                    default=default_groups,
                    key=f"resp_selected_groups_{selected_group_col}",
                )
                group_order = selected_groups if selected_groups else None

            if not selected_groups:
                st.info("Select at least one group.")
                return

            group_col_to_use = selected_group_col
        else:
            group_col_to_use = None

    st.divider()


    # ========================================================================
    # RELIABILITY CHECK (optional but recommended)
    # ========================================================================
    st.subheader("🔍 Reliability Check")

    check_reliability = st.checkbox("Check t1-t2 stability before running analysis",
        value=False,
        key="check_reliability",
        help="Verify that your reliability measurements (t1, t2) are stable enough for TE estimation",
    )

    if check_reliability:
        stability_result = check_t1_t2_stability(df, t1_col, t2_col )

        if stability_result and stability_result['verdict'] == 'unstable':
            st.error("⚠️ WARNING: Your reliability measurements show systematic change or poor stability.")
            st.warning("Proceeding with TE estimation may not be valid. Consider:")
            st.markdown("""
            - Using a longer stabilization period between t1 and t2
            - Checking for assay/instrument drift
            - Acknowledging this limitation in your methods section
            """)

            continue_anyway = st.checkbox("I understand the risks - continue anyway", key="continue_anyway")
            if not continue_anyway:
                st.stop()

    st.divider()

    # ========================================================================
    # RUN ANALYSIS and VISUALIZATION
    # ========================================================================
    st.subheader("3️⃣ Run Responder Analysis and Plot 📊 ")

    run_analysis = st.button("▶️ Calculate Responder Classifications", type="primary", width="stretch")

    if not run_analysis:
        st.info("Click the button above to run the analysis.")
        return

    try:
        with st.spinner("Calculating SWC, TE, and classifications..."):
            summary_df, individual_df, fig, ax = calculate_swc_te(
                df=df,
                group_col=group_col_to_use,
                id_col=id_col,
                pre_col=pre_col,
                post_col=post_col,
                t1_col=t1_col,
                t2_col=t2_col,
                threshold_type = threshold_type,
                swc_multiplier=swc_multiplier,
                ci_level=ci_level,
                higher_is_better=higher_is_better,
                use_relative=use_relative,
                error_metric = error_metric,  # "TE" or "SEM"
                plot=True,
                group_order=group_order,
                sort_within_group=True,
            )



        st.pyplot(fig, width="stretch")

        # ====================================================================
        # SUMMARY STATISTICS and CLASSIFICATION DISTRIBUTION
        # ====================================================================
        with st.expander("📈 Summary Statistics --- 👥 Responder Distribution"):

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Group-level Summary**")
                summary_display_cols = [
                    "analysis_group",
                    "n_intervention",
                    "n_retest",
                    "pre_mean",
                    "pre_sd",
                    "swc_multiplier",
                    "swc",
                    "te_change",
                    "ci_level",
                    "analysis_type",
                ]
                st.dataframe(
                    summary_df[summary_display_cols].round(4),
                    width="stretch",
                    hide_index=True,
                )

            with col2:
                st.markdown("**TE and Reliability**")
                te_display_cols = [
                    "analysis_group",
                    "n_retest",
                    "retest_difference_mean",
                    "retest_difference_sd",
                    "te_change",
                ]
                st.dataframe(
                    summary_df[te_display_cols].round(4),
                    width="stretch",
                    hide_index=True,
                )
            st.divider()


            st.subheader("👥 Responder Distribution")
            class_counts = individual_df["classification"].value_counts()
            class_order_display = ["reliable_meaningful_responder", "potential_responder", "non_responder"]

            class_labels = {
                "reliable_meaningful_responder": "🟢 Reliable Meaningful Responders",
                "potential_responder": "🟡 Potential Responders",
                "non_responder": "🔴 Non-responders",
            }

            cols = st.columns(3)
            for i, cls in enumerate(class_order_display):
                with cols[i]:
                    count = class_counts.get(cls, 0)
                    pct = (count / len(individual_df) * 100) if len(individual_df) > 0 else 0
                    st.metric(label=class_labels[cls], value=f"{count} ({pct:.1f}%)")

            st.divider()


        st.divider()

        # ====================================================================
        # DOWNLOAD OPTIONS
        # ====================================================================
        st.subheader("💾 Download Results")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col2:
            csv_individual = io.BytesIO()
            individual_df.to_csv(csv_individual, index=False)
            csv_individual.seek(0)
            st.download_button(
                label="📥 Responder Individual Results (CSV)",
                data=csv_individual,
                file_name=f"responder_individual_swc{swc_multiplier}_ci{ci_level:.2f}.csv",
                mime="text/csv",
                width="stretch",
            )

        with dl_col3:
            csv_summary = io.BytesIO()
            summary_df.to_csv(csv_summary, index=False)
            csv_summary.seek(0)
            st.download_button(
                label="📥 Summary Stats (CSV)",
                data=csv_summary,
                file_name=f"responder_summary_swc{swc_multiplier}_ci{ci_level:.2f}.csv",
                mime="text/csv",
                width="stretch",
            )

        with dl_col1:
            png_buffer = fig_to_png_buffer(fig)
            st.download_button(
                label="📥 Responder Plot (PNG)",
                data=png_buffer,
                file_name=f"responder_plot_swc{swc_multiplier}_ci{ci_level:.2f}.png",
                mime="image/png",
                width="stretch",
            )

    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        st.exception(e)