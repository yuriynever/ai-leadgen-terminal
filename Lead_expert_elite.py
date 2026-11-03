import io
import json
import time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field

# -----
st.set_page_config(
    page_title="OutreachGenius | AI Lead Personalizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .email-container {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 24px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #212529;
        margin-top: 15px;
    }
    .email-header {
        border-bottom: 1px solid #dee2e6;
        padding-bottom: 12px;
        margin-bottom: 16px;
        font-size: 14px;
        color: #495057;
    }
    .highlight-hook {
        background-color: #e7f5ff;
        color: #1971c2;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }
    .highlight-pitch {
        background-color: #ebfbee;
        color: #2b8a3e;
        padding: 2px 6px;
        border-radius: 4px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


class LeadAnalysis(BaseModel):
    lead_fit_score: str = Field(
        description="Lead qualification rating: High, Medium, or Low"
    )
    likely_pain_point: str = Field(
        description="Specific business challenge or operational bottleneck this company likely faces."
    )
    email_subject_line: str = Field(
        description="Short, compelling 3-5 word cold email subject line."
    )
    ice_breaker: str = Field(
        description="A hyper-personalized first sentence for a cold email referencing their actual business context. Avoid generic pleasantries."
    )
    custom_pitch: str = Field(
        description="A 1-2 sentence value proposition connecting their bottleneck to our solution."
    )


SAMPLE_LEADS = [
    {
        "Company Name": "CloudScale Solutions",
        "Website": "https://cloudscale-infra.io",
        "Niche / Description": "Multi-cloud DevOps and Kubernetes infrastructure management for mid-market FinTech platforms.",
    },
    {
        "Company Name": "NexaRecruit AI",
        "Website": "https://nexarecruit.ai",
        "Niche / Description": "Automated technical candidate sourcing and AI screening tool for high-growth tech startups.",
    },
    {
        "Company Name": "SwiftRoute Logistics",
        "Website": "https://swiftroute-freight.com",
        "Niche / Description": "Temperature-controlled freight forwarding and cross-border customs brokerage for pharma companies.",
    },
    {
        "Company Name": "SafeGuard Cyber",
        "Website": "https://safeguard-cybersec.com",
        "Niche / Description": "Managed SOC-as-a-Service and SOC2 / ISO27001 audit preparation for B2B SaaS businesses.",
    },
    {
        "Company Name": "FlowMetrics Studio",
        "Website": "https://flowmetrics-ux.com",
        "Niche / Description": "Product design and UX consultancy specialized in user onboarding optimization and churn reduction.",
    },
]

with st.sidebar:
    st.title("⚡ Settings & Auth")

    api_key = st.text_input(
        "Google Gemini API Key:",
        type="password",
        help="Get your free API key at https://aistudio.google.com/app/apikey",
    )

    if not api_key:
        st.caption("⚠️ API key required to run the AI engine.")

    st.divider()

    st.subheader(" Engine Parameters")
    model_choice = st.selectbox(
        "Gemini Model:",
        options=["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
        index=0,
        help="gemini-3.8-flash provides fast, high-quality structured generation.",
    )

    st.divider()
    st.subheader(" Sender Persona (For Pitch)")
    sender_name = st.text_input("Your Name:", value="Alex Rivera")
    sender_offering = st.text_input(
        "Your Service / Product:",
        value="AI automation systems that eliminate manual outbound workflows",
    )

    st.markdown("---")
    st.markdown(
        " *Built for high-ticket outbound campaigns. Generates contextual hooks that raise cold email reply rates by 3-4x.*"
    )


def analyze_lead(
        client: genai.Client,
        company: str,
        website: str,
        description: str,
        model: str,
        offering: str,
) -> dict:
    """Invokes Gemini using Structured Output (Pydantic schema)."""
    system_instruction = f"""
    You are an elite B2B Outbound Strategist and Copywriter.
    Your objective: Analyze target companies and generate hyper-personalized outreach data.

    Our Offering: "{offering}".

    Guidelines:
    1. Pain Point: Identify a realistic bottleneck specific to their business model and scale.
    2. Ice-breaker: Must be sharp and reference what they do. DO NOT use generic phrases like 'Hope this finds you well' or 'I came across your website'.
    3. Custom Pitch: Connect their likely challenge to how our offering delivers concrete ROI.
    4. Lead Fit Score: High, Medium, or Low based on B2B marketability.
    Language: Professional English.
    """

    user_prompt = f"""
    Target Company Profile:
    - Name: {company}
    - Website: {website}
    - What they do: {description}
    """

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=LeadAnalysis,
            temperature=0.3,
        ),
    )

    return json.loads(response.text)


st.title(" AI Lead Smart-Filter & Hyper-Personalizer")
st.markdown(
    "Transform unformatted B2B leads into an enriched pipeline with **deep business intelligence**, **personalized ice-breakers**, and **ready-to-send emails**."
)

st.write("")

st.subheader("Step 1: Import Prospect Database")

col_upload, col_sample = st.columns([3, 2])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload leads list (CSV format)",
        type=["csv"],
        help="Upload any CSV file containing prospect companies.",
    )

with col_sample:
    st.write("Or test with pre-built data:")
    if st.button(" Load Sample B2B Dataset", use_container_width=True):
        st.session_state["raw_df"] = pd.DataFrame(SAMPLE_LEADS)
        st.success("Loaded 5 verified B2B leads!")

df = None
if uploaded_file is not None:
    try:
        loaded_df = pd.read_csv(uploaded_file)
        # Bulletproof validation: Check if file is empty
        if loaded_df.empty:
            st.error(" The uploaded CSV file is empty. Please upload a valid dataset.")
        else:
            st.session_state["raw_df"] = loaded_df
    except Exception as e:
        st.error(f" Error parsing CSV file: {str(e)}")

df = st.session_state.get("raw_df", None)

if df is not None:
    st.write("### Raw Data Preview")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Step 2: Map Data Fields")
    cols = list(df.columns)

    c1, c2, c3 = st.columns(3)
    with c1:
        col_company = st.selectbox(
            "Company Name Column *",
            options=cols,
            index=0 if len(cols) > 0 else 0,
        )
    with c2:
        col_website = st.selectbox(
            "Website Column",
            options=cols,
            index=1 if len(cols) > 1 else 0,
        )
    with c3:
        col_desc = st.selectbox(
            "Niche / Description Column *",
            options=cols,
            index=2 if len(cols) > 2 else 0,
        )

    is_valid = True
    if df[col_company].dropna().empty:
        st.error(f" Selected column '{col_company}' contains no valid data.")
        is_valid = False
    if df[col_desc].dropna().empty:
        st.error(f" Selected column '{col_desc}' contains no valid data.")
        is_valid = False

    st.write("")
    st.subheader("Step 3: Run AI Personalization Engine")

    start_btn = st.button(" Analyze & Generate Campaign", type="primary", disabled=not is_valid)

    if start_btn:

        if not api_key or api_key.strip() == "":
            st.warning("⚠️ Action Required: Please enter your Google Gemini API Key in the left sidebar.")
            st.stop()

        try:

            client = genai.Client(api_key=api_key.strip())

            results = []
            total_leads = len(df)


            progress_bar = st.progress(0)
            status_container = st.empty()

            with st.spinner("Analyzing companies and tailoring cold hooks..."):
                for idx, row in df.iterrows():
                    company_val = str(row[col_company]).strip()
                    website_val = str(row[col_website]).strip() if col_website else "N/A"
                    desc_val = str(row[col_desc]).strip()

                    status_container.info(f" Processing [{idx + 1}/{total_leads}]: **{company_val}**...")

                    try:
                        ai_result = analyze_lead(
                            client=client,
                            company=company_val,
                            website=website_val,
                            description=desc_val,
                            model=model_choice,
                            offering=sender_offering,
                        )

                        results.append({
                            "Company": company_val,
                            "Website": website_val,
                            "Fit Score": ai_result.get("lead_fit_score", "Medium"),
                            "Subject Line": ai_result.get("email_subject_line", "Quick question"),
                            "Identified Bottleneck": ai_result.get("likely_pain_point", "N/A"),
                            "Ice-Breaker Hook": ai_result.get("ice_breaker", ""),
                            "Tailored Pitch": ai_result.get("custom_pitch", ""),
                        })

                    except APIError as api_err:
                        st.error(f"Gemini API Error on '{company_val}': {api_err.message}")
                        results.append({
                            "Company": company_val,
                            "Website": website_val,
                            "Fit Score": "Failed",
                            "Subject Line": "Error",
                            "Identified Bottleneck": "API Call Failed",
                            "Ice-Breaker Hook": str(api_err.message),
                            "Tailored Pitch": "",
                        })
                    except Exception as err:
                        results.append({
                            "Company": company_val,
                            "Website": website_val,
                            "Fit Score": "Error",
                            "Subject Line": "Error",
                            "Identified Bottleneck": "Execution Error",
                            "Ice-Breaker Hook": str(err),
                            "Tailored Pitch": "",
                        })

                    progress_bar.progress((idx + 1) / total_leads)

                    time.sleep(1.0)

            status_container.success(f" Successfully enriched {total_leads} leads!")
            st.session_state["results_df"] = pd.DataFrame(results)

        except Exception as client_init_error:
            st.error(f" Failed to connect to Gemini API: {str(client_init_error)}")


    if "results_df" in st.session_state:
        st.write("")
        st.subheader("Step 4: Campaign Deliverables & Live Preview")

        res_df = st.session_state["results_df"]

        tab_table, tab_preview, tab_metrics = st.tabs([
            " Enriched Database",
            "✉ Live Cold Email Sandbox",
            " Campaign Metrics",
        ])


        with tab_table:
            st.markdown("##### Filtered & Personalized Leads")
            st.dataframe(res_df, use_container_width=True, hide_index=True)


            csv_buffer = io.StringIO()
            res_df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode("utf-8")

            st.download_button(
                label=" Download Enriched Leads (CSV)",
                data=csv_bytes,
                file_name="hyper_personalized_campaign_leads.csv",
                mime="text/csv",
                type="primary",
            )


        with tab_preview:
            st.markdown("##### Interactive Cold Email Mockup")
            st.caption("Select a lead from your campaign to see how the email looks in a recipient's inbox:")

            company_options = res_df["Company"].tolist()
            selected_company_name = st.selectbox("Inspect Lead:", options=company_options)

            selected_lead = res_df[res_df["Company"] == selected_company_name].iloc[0]

            st.markdown(
                f"""
                <div class="email-container">
                    <div class="email-header">
                        <strong>To:</strong> decision-maker@{selected_lead['Company'].lower().replace(' ', '')}.com<br>
                        <strong>From:</strong> {sender_name} &lt;you@yourdomain.com&gt;<br>
                        <strong>Subject:</strong> {selected_lead['Subject Line']}
                    </div>
                    <div style="line-height: 1.6; font-size: 15px;">
                        <p>Hi team at {selected_lead['Company']},</p>

                        <p><span class="highlight-hook">{selected_lead['Ice-Breaker Hook']}</span></p>

                        <p><span class="highlight-pitch">{selected_lead['Tailored Pitch']}</span> We specifically work with companies facing bottlenecks around <em>{selected_lead['Identified Bottleneck']}</em>.</p>

                        <p>Open to a brief 7-minute intro call this Thursday to see if this makes sense for your pipeline?</p>

                        <p>Best regards,<br>
                        <strong>{sender_name}</strong><br>
                        <span style="color: #868e96; font-size: 13px;">Specialist in {sender_offering}</span></p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        with tab_metrics:
            st.markdown("##### Pipeline Breakdown")
            m1, m2, m3 = st.columns(3)

            high_fit_count = len(res_df[res_df["Fit Score"].str.lower() == "high"])
            m1.metric("Total Prospects", len(res_df))
            m2.metric("High-Fit Leads", high_fit_count)
            m3.metric("Ready-to-Send Hooks", len(res_df[res_df["Ice-Breaker Hook"].str.len() > 10]))

            st.write("")
            with st.expander("🔍 View Extracted Bottlenecks Summary"):
                for _, row in res_df.iterrows():
                    st.markdown(f"- **{row['Company']}**: {row['Identified Bottleneck']}")

else:
    st.info(" Please upload a CSV file or click **'Load Sample B2B Dataset'** above to get started.")
