# flake8: noqa: E501
"""
Streamlit app for TOO GOOD TO GO Pitch Maker
This app allows users to search for food businesses using Google Places API
and generate pitch decks for TOO GOOD TO GO sales representatives.
"""

import io
import json
import logging
import os
from contextlib import redirect_stdout
from typing import List

import streamlit as st
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from main import create_pitch_deck, print_pitch_deck, search_places

# Load environment variables
load_dotenv()


# Pydantic model for TGTG relevance assessment
class TGTGRelevanceAssessment(BaseModel):
    relevance: int = Field(
        description="Relevance score from 0-10 for TOO GOOD TO GO partnership potential",
        ge=0,
        le=10,
    )
    reason: str = Field(
        description="Brief explanation of the relevance score focusing on TGTG partnership potential"
    )


class FoodBusinessRanking(BaseModel):
    assessments: List[TGTGRelevanceAssessment] = Field(
        description="List of relevance assessments for each food business in the same order as provided"
    )


def evaluate_food_business_relevance(
    food_businesses: List[dict],
) -> List[TGTGRelevanceAssessment]:
    """
    Evaluate the TGTG partnership potential of a list of food businesses using LLM.

    Args:
        food_businesses: List of food business data from Google Places API

    Returns:
        List of TGTGRelevanceAssessment objects with relevance scores and reasons
    """
    if not food_businesses:
        return []

    # Initialize the Google Gemini API client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found. Cannot evaluate relevance.")
        return []

    client = genai.Client(api_key=api_key)

    # Prepare simplified business data for the LLM
    simplified_businesses = []
    for business in food_businesses:
        simplified = {
            "name": business.get("displayName", {}).get("text", "Unknown"),
            "address": business.get("formattedAddress", ""),
            "types": business.get("types", []),
            "rating": business.get("rating"),
            "user_rating_count": business.get("userRatingCount"),
            "price_level": business.get("priceLevel"),
            "business_status": business.get("businessStatus"),
            "services": {
                "delivery": business.get("delivery"),
                "dine_in": business.get("dineIn"),
                "takeout": business.get("takeout"),
            },
        }
        simplified_businesses.append(simplified)

    # Create the prompt for TGTG relevance assessment
    assessment_prompt = f"""
    ## CONTEXT: TOO GOOD TO GO BUSINESS MODEL
    TOO GOOD TO GO is a marketplace that connects consumers with food businesses that have surplus food.
    Businesses sell surplus food at reduced prices through the app, reducing waste while generating revenue.

    IDEAL PARTNERS have these characteristics:
    - High food waste potential (daily fresh items, baked goods, prepared meals)
    - Regular operating schedule with predictable surplus
    - Multiple meal times or high product variety
    - Good reputation (ratings) and customer base
    - Openness to sustainability initiatives
    - Delivery/takeout capability preferred

    ## TASK
    Evaluate each food business below for TOO GOOD TO GO partnership potential.
    Score each from 0-10 (10 = perfect fit) and provide reasoning.

    ## FOOD BUSINESSES TO EVALUATE:
    ```json
    {json.dumps(simplified_businesses, indent=2)}
    ```

    ## SCORING CRITERIA:
    - **10-9**: Perfect fit (bakeries, cafes with fresh items, restaurants with daily specials)
    - **8-7**: Very good fit (most restaurants, delis, food retailers with perishables)
    - **6-5**: Moderate fit (basic restaurants, some retail food)
    - **4-3**: Low fit (fast food chains, limited fresh items)
    - **2-1**: Poor fit (non-food or very limited waste potential)
    - **0**: Not suitable (no food waste potential)

    Focus on: food waste potential, business type, operating patterns, and sustainability alignment.
    """

    try:
        # Generate the structured assessment
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=assessment_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FoodBusinessRanking,
            },
        )

        if hasattr(response, "parsed"):
            ranking = response.parsed
            return ranking.assessments
        else:
            # Fallback to manual JSON parsing
            ranking_data = json.loads(response.text)
            assessments = []
            for assessment_data in ranking_data.get("assessments", []):
                assessment = TGTGRelevanceAssessment(
                    relevance=assessment_data.get("relevance", 0),
                    reason=assessment_data.get("reason", "No assessment available"),
                )
                assessments.append(assessment)
            return assessments

    except Exception as e:
        st.error(f"Error evaluating relevance: {str(e)}")
        # Return default assessments if LLM fails
        return [
            TGTGRelevanceAssessment(
                relevance=5, reason="Assessment unavailable due to technical error"
            )
            for _ in food_businesses
        ]


# Configure page
st.set_page_config(
    page_title="TOO GOOD TO GO Pitch Maker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.toogoodtogo.com",
        "Report a bug": None,
        "About": "TOO GOOD TO GO Pitch Maker - Helping reduce food waste through strategic partnerships",
    },
)

# Custom CSS for TOO GOOD TO GO brand styling
st.markdown(
    """
<style>
    /* TOO GOOD TO GO brand colors: green (#00D68F) and dark green (#005A2D) */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8fffe 100%);
    }

    .main-header {
        background: linear-gradient(135deg, #00D68F 0%, #00B377 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 214, 143, 0.15);
        margin-bottom: 2rem;
        text-align: center;
    }

    .main-header h1 {
        color: white !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        font-size: 1.1rem;
        margin-bottom: 0;
    }

    .section-header {
        background: linear-gradient(135deg, #00D68F 0%, #00B377 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #005A2D;
    }

    .section-header h2, .section-header h3 {
        color: white !important;
        margin-bottom: 0.5rem;
    }

    .section-header p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin-bottom: 0;
    }

    .result-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 214, 143, 0.1);
        border: 1px solid #e8f6f3;
    }

    .log-container {
        background: #f8fffe;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.9em;
        max-height: 400px;
        overflow-y: auto;
        margin: 1rem 0;
        border: 1px solid #e8f6f3;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #f8fffe;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00D68F 0%, #00B377 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00B377 0%, #009960 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 214, 143, 0.3);
    }

    /* Form styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e8f6f3;
        transition: border-color 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00D68F;
        box-shadow: 0 0 0 2px rgba(0, 214, 143, 0.2);
    }

    /* Metric styling */
    .css-1xarl3l {
        background: #f8fffe;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e8f6f3;
    }

    /* Success/Info messages */
    .stSuccess {
        background: #e8f6f3;
        border: 1px solid #00D68F;
        color: #005A2D;
    }

    .stInfo {
        background: #e8f6f3;
        border: 1px solid #00D68F;
        color: #005A2D;
    }

    /* Radio button styling */
    .stRadio > div {
        background: #f8fffe;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e8f6f3;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: #00D68F;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8fffe;
        border: 1px solid #e8f6f3;
        border-radius: 8px;
    }

    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #005A2D 0%, #004225 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #004225 0%, #00331c 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 90, 45, 0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🍽️ TOO GOOD TO GO Pitch Maker</h1>
        <p>Discover restaurants and create compelling pitch decks for sustainable partnerships</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Choose a function:",
        ["🔍 Food Business Search", "📊 Pitch Deck Generator"],
        index=0,
    )

    if page == "🔍 Food Business Search":
        food_business_search_page()
    elif page == "📊 Pitch Deck Generator":
        pitch_deck_page()


def food_business_search_page():
    st.markdown(
        """
    <div class="section-header">
        <h2>🔍 Food Business Search</h2>
        <p>Search for food businesses and stores in any location using Google Places API</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Input form
    with st.form("search_form"):
        st.subheader("Enter Search Query")
        query = st.text_input(
            "Search for food businesses:",
            placeholder="e.g., Chinese restaurant in Taufkirchen, Bavaria, Germany",
            help="Enter a search query like 'Italian restaurant in Munich' or 'Asian supermarket in Berlin'",
        )

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            st.info("The following essential fields are included in the search:")
            st.write("• Food business name and address")
            st.write("• Contact information (phone, website)")
            st.write("• Ratings and reviews")
            st.write("• Basic services (delivery, dine-in, takeout)")
            st.write("• Cuisine type and price level")

        submitted = st.form_submit_button(
            "🔍 Search Food Businesses", use_container_width=True
        )

    if submitted and query:
        search_food_businesses(query)


def search_food_businesses(query):
    # Define essential fields for restaurant search (to avoid pagination issues)
    fields_to_extract = [
        "places.displayName",
        "places.formattedAddress",
        "places.businessStatus",
        "places.types",
        "places.primaryType",
        "places.rating",
        "places.userRatingCount",
        "places.websiteUri",
        "places.internationalPhoneNumber",
        "places.priceLevel",
        "places.delivery",
        "places.dineIn",
        "places.takeout",
    ]

    with st.spinner("🔍 Searching for food businesses..."):
        try:
            results = search_places(query, fields_to_extract)

            if isinstance(results, dict) and "error" in results:
                st.error(f"❌ Error: {results['error']}")
                return

            if not results:
                st.warning("No food businesses found for your search query.")
                return

            # Display results
            st.markdown(
                """
            <div class="section-header">
                <h3>📍 Search Results</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Evaluate TGTG relevance and rank results
            with st.spinner("🤖 Evaluating TGTG partnership potential..."):
                assessments = evaluate_food_business_relevance(results)

            # Combine results with their assessments and sort by relevance
            if assessments and len(assessments) == len(results):
                combined_results = list(zip(results, assessments))
                # Sort by relevance score (highest first)
                combined_results.sort(key=lambda x: x[1].relevance, reverse=True)
                results, assessments = zip(*combined_results)
                results = list(results)
                assessments = list(assessments)

            st.info(
                f"Found {len(results)} food business(es) matching your search, ranked by TGTG partnership potential."
            )

            display_all_food_business_results(results, assessments)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


def display_all_food_business_results(food_businesses, assessments=None):
    """Display all food business results with basic information in a list format"""
    for i, food_business in enumerate(food_businesses, 1):
        st.markdown('<div class="result-container">', unsafe_allow_html=True)

        # Food business name and basic info
        name = food_business.get("displayName", {}).get("text", "Unknown Food Business")
        address = food_business.get("formattedAddress", "Address not available")

        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            # Create ranking display with medals and TGTG relevance
            ranking_display = ""
            assessment = (
                assessments[i - 1] if assessments and i - 1 < len(assessments) else None
            )

            if assessment:
                # Medal display for top 3
                if i == 1:
                    medal = "🥇"
                elif i == 2:
                    medal = "🥈"
                elif i == 3:
                    medal = "🥉"
                else:
                    medal = f"{i}."

                # Create ranking with tooltip
                relevance_score = assessment.relevance
                reason = assessment.reason

                # Color code based on relevance score
                if relevance_score >= 8:
                    score_color = "#00D68F"  # TGTG green for high relevance
                elif relevance_score >= 6:
                    score_color = "#FFA726"  # Orange for medium relevance
                else:
                    score_color = "#78909C"  # Gray for low relevance

                # Create ranking display with better tooltip using Streamlit columns
                rank_col1, rank_col2 = st.columns([1, 4])

                with rank_col1:
                    st.markdown(
                        f"<span style='font-size: 1.5em;'>{medal}</span>",
                        unsafe_allow_html=True,
                    )

                with rank_col2:
                    # Use Streamlit metric with help parameter for tooltip
                    st.metric(
                        label="TGTG Relevance",
                        value=f"{relevance_score}/10",
                        help=f"💡 Assessment: {reason}",
                        delta=None,
                    )

                    # Add color styling with CSS
                    if relevance_score >= 8:
                        badge_style = "🟢 High Priority"
                    elif relevance_score >= 6:
                        badge_style = "🟡 Medium Priority"
                    else:
                        badge_style = "⚪ Low Priority"

                    st.caption(badge_style)
            else:
                st.markdown(f"### {i}.")

            st.markdown(f"### 🏪 {name}")
            st.markdown(f"📍 {address}")

            # Cuisine type
            if food_business.get("types"):
                types = food_business["types"]
                cuisine_types = [
                    t.replace("_", " ").title()
                    for t in types
                    if "restaurant" in t.lower() or "food" in t.lower()
                ]
                if cuisine_types:
                    st.markdown(f"🍽️ **Type:** {', '.join(cuisine_types[:2])}")

        with col2:
            # Rating and reviews
            if food_business.get("rating"):
                rating = food_business["rating"]
                count = food_business.get("userRatingCount", 0)
                st.metric("⭐ Rating", f"{rating}/5")
                st.caption(f"{count} reviews")

            # Business status
            if food_business.get("businessStatus"):
                status = food_business["businessStatus"].replace("_", " ").title()
                if status == "Operational":
                    st.success("✅ Open")
                else:
                    st.warning(f"⚠️ {status}")

        with col3:
            # Contact info
            if food_business.get("internationalPhoneNumber"):
                st.markdown(f"📞 {food_business['internationalPhoneNumber']}")

            if food_business.get("websiteUri"):
                st.markdown(f"[🌐 Website]({food_business['websiteUri']})")

            # Price level
            if food_business.get("priceLevel"):
                try:
                    price_num = int(food_business["priceLevel"])
                    price_level = "💰" * price_num
                    st.markdown(f"**Price:** {price_level}")
                except (ValueError, TypeError):
                    st.markdown(f"**Price:** {food_business['priceLevel']}")

        # Quick features summary
        features = []
        if food_business.get("delivery"):
            features.append("🚚 Delivery")
        if food_business.get("dineIn"):
            features.append("🍽️ Dine-in")
        if food_business.get("takeout"):
            features.append("🥡 Takeout")

        if features:
            st.markdown(f"**Services:** {' • '.join(features)}")

        st.markdown("</div>", unsafe_allow_html=True)


def display_restaurant_result(restaurant):
    st.markdown('<div class="result-container">', unsafe_allow_html=True)

    # Restaurant name and basic info
    name = restaurant.get("displayName", {}).get("text", "Unknown Restaurant")
    address = restaurant.get("formattedAddress", "Address not available")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### 🏪 {name}")
        st.markdown(f"📍 **Address:** {address}")

        # Contact information
        if restaurant.get("internationalPhoneNumber"):
            st.markdown(f"📞 **Phone:** {restaurant['internationalPhoneNumber']}")

        if restaurant.get("websiteUri"):
            st.markdown(f"🌐 **Website:** {restaurant['websiteUri']}")

    with col2:
        # Rating and reviews
        if restaurant.get("rating"):
            rating = restaurant["rating"]
            count = restaurant.get("userRatingCount", 0)
            st.metric("⭐ Rating", f"{rating}/5", f"{count} reviews")

        # Price level
        if restaurant.get("priceLevel"):
            try:
                price_num = int(restaurant["priceLevel"])
                price_level = "💰" * price_num
                st.metric("💰 Price Level", price_level)
            except (ValueError, TypeError):
                st.metric("💰 Price Level", restaurant["priceLevel"])

    # Restaurant features
    st.markdown("#### 🍽️ Restaurant Features")
    features = []

    if restaurant.get("delivery"):
        features.append("🚚 Delivery")
    if restaurant.get("dineIn"):
        features.append("🍽️ Dine-in")
    if restaurant.get("takeout"):
        features.append("🥡 Takeout")
    if restaurant.get("outdoorSeating"):
        features.append("🌤️ Outdoor Seating")

    if features:
        st.write(" • ".join(features))

    # Meal services
    meals = []
    if restaurant.get("servesBreakfast"):
        meals.append("🌅 Breakfast")
    if restaurant.get("servesLunch"):
        meals.append("🌞 Lunch")
    if restaurant.get("servesDinner"):
        meals.append("🌙 Dinner")
    if restaurant.get("servesVegetarianFood"):
        meals.append("🥗 Vegetarian")

    if meals:
        st.markdown("**Meals:** " + " • ".join(meals))

    # Opening hours
    if restaurant.get("currentOpeningHours") or restaurant.get("regularOpeningHours"):
        with st.expander("🕐 Opening Hours"):
            hours = restaurant.get("currentOpeningHours") or restaurant.get(
                "regularOpeningHours"
            )
            st.json(hours)

    # Raw data
    with st.expander("🔍 Raw Data"):
        st.json(restaurant)

    st.markdown("</div>", unsafe_allow_html=True)


def pitch_deck_page():
    st.markdown(
        """
    <div class="section-header">
        <h2>📊 Pitch Deck Generator</h2>
        <p>Generate comprehensive pitch decks for TOO GOOD TO GO sales representatives</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Input form
    with st.form("pitch_form"):
        st.subheader("Food Business Information")
        query = st.text_input(
            "Enter food business query:",
            placeholder="e.g., Help me prepare a pitch deck for Pak Choi in Taufkirchen, Bavaria, Germany",
            help="Describe the food business you want to create a pitch deck for",
        )

        st.info(
            "💡 The pitch deck will include contact information, decision maker profile, key statistics, and tailored pitch strategies."
        )

        submitted = st.form_submit_button(
            "🚀 Generate Pitch Deck", use_container_width=True
        )

    if submitted and query:
        generate_pitch_deck(query)


class StreamlitLogHandler(logging.Handler):
    """Custom log handler that streams to Streamlit in real-time"""

    def __init__(self, log_container):
        super().__init__()
        self.log_container = log_container
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        # Update the display with all logs in a scrollable container
        formatted_logs = "\n".join(self.logs)
        self.log_container.text_area(
            "Processing Logs",
            value=formatted_logs,
            height=200,
            disabled=True,
            label_visibility="collapsed",
        )


def generate_pitch_deck(query):
    # Create streaming log display
    with st.status("🚀 Generating pitch deck...", expanded=True) as status:
        log_container = st.empty()

        # Create custom log handler for streaming
        log_handler = StreamlitLogHandler(log_container)
        log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S"
        )
        log_handler.setFormatter(formatter)

        # Get the logger from main.py
        logger = logging.getLogger("pitchmaker")
        logger.addHandler(log_handler)

        try:
            # Generate pitch deck with streaming logs
            pitch_deck = create_pitch_deck(query)

            # Remove the log handler
            logger.removeHandler(log_handler)

            # Update status
            status.update(
                label="✅ Pitch deck generated successfully!", state="complete"
            )

            # Display pitch deck
            if "error" in pitch_deck:
                st.error(f"❌ Error generating pitch deck: {pitch_deck['error']}")
                if "raw_response" in pitch_deck:
                    with st.expander("🔍 Raw Response"):
                        st.text(pitch_deck["raw_response"])
            else:
                display_pitch_deck(pitch_deck)

        except Exception as e:
            logger.removeHandler(log_handler)
            status.update(label="❌ Error occurred during generation", state="error")
            st.error(f"❌ An error occurred: {str(e)}")


def display_pitch_deck(pitch_deck):
    st.markdown(
        """
    <div class="section-header">
        <h3>📊 Generated Pitch Deck</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    contact = pitch_deck.get("contact_info", {})

    # Contact Information
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("📞 Contact Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Restaurant:** {contact.get('name', 'N/A')}")
        st.write(f"**Address:** {contact.get('address', 'N/A')}")
        if contact.get("phone"):
            st.write(f"**Phone:** {contact['phone']}")
        if contact.get("website"):
            st.write(f"**Website:** {contact['website']}")

    with col2:
        if contact.get("cuisine_type"):
            st.write(f"**Cuisine:** {', '.join(contact['cuisine_type'])}")

        # Format opening hours properly
        if contact.get("opening_hours"):
            st.write("**Opening Hours:**")
            hours_data = contact["opening_hours"]

            # Handle different formats of opening hours
            if isinstance(hours_data, dict) and "weekdayDescriptions" in hours_data:
                # Display each day on its own line, properly formatted
                for day_info in hours_data["weekdayDescriptions"]:
                    clean_day = " ".join(day_info.split())  # Clean up whitespace
                    st.caption(clean_day)
            elif isinstance(hours_data, str):
                # For string format, split by semicolons and display each day
                days = hours_data.split(";")
                for day in days:
                    clean_day = " ".join(day.strip().split())  # Clean up whitespace
                    if clean_day:
                        st.caption(clean_day)
            else:
                st.caption("Hours available on request")

    st.markdown("</div>", unsafe_allow_html=True)

    # Recommended Approach (moved right after contact info for logical flow)
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("💡 Recommended Approach")

    approach_text = pitch_deck.get("recommended_approach", "No approach provided")
    st.write(approach_text)

    # Lead assessment info with vivid temperature display
    col1, col2 = st.columns(2)
    with col1:
        lead_temp = pitch_deck.get("lead_temperature", "unknown").lower()
        lead_temp_help = pitch_deck.get(
            "lead_temperature_reasoning",
            "Assessment based on food business characteristics and likelihood to convert",
        )

        # Create vivid temperature display with colors and icons
        if lead_temp == "hot":
            st.markdown(
                "<div style='background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>"
                "<h3 style='margin: 0; color: white;'>🔥🔥🔥 HOT LEAD 🔥🔥🔥</h3>"
                "<p style='margin: 0.5rem 0 0 0; color: white;'>High conversion probability</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif lead_temp == "warm":
            st.markdown(
                "<div style='background: linear-gradient(135deg, #FFA726 0%, #FFB74D 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>"
                "<h3 style='margin: 0; color: white;'>🔥🔥 WARM LEAD 🔥🔥</h3>"
                "<p style='margin: 0.5rem 0 0 0; color: white;'>Moderate conversion probability</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif lead_temp == "cold":
            st.markdown(
                "<div style='background: linear-gradient(135deg, #42A5F5 0%, #64B5F6 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>"
                "<h3 style='margin: 0; color: white;'>❄️ COLD LEAD ❄️</h3>"
                "<p style='margin: 0.5rem 0 0 0; color: white;'>Lower conversion probability</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background: linear-gradient(135deg, #78909C 0%, #90A4AE 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white; font-weight: bold;'>"
                "<h3 style='margin: 0; color: white;'>❔ UNKNOWN LEAD ❔</h3>"
                "<p style='margin: 0.5rem 0 0 0; color: white;'>Assessment pending</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Show reasoning in a smaller text below
        if lead_temp_help:
            st.caption(f"📝 Reasoning: {lead_temp_help}")
    with col2:
        if pitch_deck.get("best_contact_time"):
            st.markdown(
                "<div style='background: linear-gradient(135deg, #00D68F 0%, #00B377 100%); padding: 1rem; border-radius: 10px; color: white;'>"
                f"<h4 style='margin: 0; color: white;'>⏰ Best Contact Time</h4>"
                f"<p style='margin: 0.5rem 0 0 0; color: white;'>{pitch_deck['best_contact_time']}</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Key Statistics
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("📈 Key Statistics")

    stats = pitch_deck.get("key_stats", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if stats.get("user_rating"):
            st.metric("⭐ Rating", f"{stats['user_rating']}/5")
        else:
            st.metric("⭐ Rating", "N/A")

    with col2:
        sustainability_help = stats.get(
            "sustainability_reasoning", "Assessment based on available information"
        )
        st.metric(
            "🌱 Sustainability",
            stats.get("sustainability_signal", "unknown").title(),
            help=sustainability_help,
        )

    with col3:
        digital_help = stats.get(
            "digital_readiness_reasoning",
            "Assessment based on online presence and digital footprint",
        )
        st.metric(
            "💻 Digital Ready",
            stats.get("digital_readiness", "unknown").title(),
            help=digital_help,
        )

    with col4:
        if stats.get("user_rating_count"):
            st.metric("📊 Reviews", f"{stats['user_rating_count']}")
        else:
            st.metric("📊 Reviews", "N/A")

    # Additional stats in a more readable format
    col1, col2 = st.columns(2)
    with col1:
        if stats.get("estimated_food_waste"):
            st.write(f"**Estimated Food Waste:** {stats['estimated_food_waste']}")

    with col2:
        if stats.get("estimated_revenue_potential"):
            st.write(f"**Revenue Potential:** {stats['estimated_revenue_potential']}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Decision Maker Profile
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("👤 Decision Maker Profile")

    profile = pitch_deck.get("decision_maker_profile", {})
    if profile.get("summary"):
        st.write(f"**Summary:** {profile['summary']}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Pain Points:**")
        for point in profile.get("pain_points", []):
            st.write(f"• {point}")

    with col2:
        st.write("**Values:**")
        for value in profile.get("values", []):
            st.write(f"• {value}")

    if profile.get("communication_style"):
        st.write(f"**Communication Style:** {profile['communication_style']}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Pitch Strategy
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("🎯 Pitch Strategy")

    strategy = pitch_deck.get("pitch_strategy", {})

    if strategy.get("opening_hook"):
        st.write("**Opening Hook:**")
        st.write(strategy["opening_hook"])

    if strategy.get("value_proposition"):
        st.write("**Value Proposition:**")
        st.write(strategy["value_proposition"])

    # Display persuasion techniques with expanded behavioral science information
    if strategy.get("persuasion_techniques"):
        st.write("**🧠 Behavioral Science Techniques:**")
        for i, technique in enumerate(strategy["persuasion_techniques"], 1):
            with st.expander(f"🎯 {technique.get('technique_name', f'Technique {i}')}"):
                st.write(
                    f"**Description:** {technique.get('description', 'No description available')}"
                )
                st.write(
                    f"**Application:** {technique.get('application', 'No application details')}"
                )
                st.write(
                    f"**Why Effective:** {technique.get('effectiveness_reason', 'No reasoning provided')}"
                )

                # Highlight the ready-to-use pitch script
                if technique.get("pitch_script"):
                    st.markdown("**📝 Ready-to-Use Script:**")
                    st.info(f'"{technique["pitch_script"]}"')

    if strategy.get("urgency_closing"):
        st.write("**🔥 Closing Statement:**")
        st.write(strategy["urgency_closing"])

    # Objection handling in a more user-friendly format
    if strategy.get("objection_handling"):
        st.write("**Objection Handling:**")
        for i, obj in enumerate(strategy["objection_handling"], 1):
            with st.expander(f"❓ Objection {i}: {obj.get('objection', 'Objection')}"):
                st.write(f"**Response:** {obj.get('response', 'No response provided')}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Additional Notes
    if pitch_deck.get("additional_notes"):
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.subheader("📝 Additional Notes")
        for note in pitch_deck["additional_notes"]:
            st.write(f"• {note}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Download Options
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    st.subheader("📥 Download Options")

    col1, col2 = st.columns(2)

    with col1:
        # JSON download
        json_data = json.dumps(pitch_deck, indent=2)
        st.download_button(
            label="📄 Download as JSON",
            data=json_data,
            file_name=f"pitch_deck_{contact.get('name', 'restaurant').replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        # Text format download
        text_output = io.StringIO()
        with redirect_stdout(text_output):
            print_pitch_deck(pitch_deck)

        text_data = text_output.getvalue()
        st.download_button(
            label="📝 Download as Text",
            data=text_data,
            file_name=f"pitch_deck_{contact.get('name', 'restaurant').replace(' ', '_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
