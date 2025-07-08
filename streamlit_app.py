# flake8: noqa: E501
"""
Streamlit app for TOO GOOD TO GO Pitch Maker
This app allows users to search for restaurants using Google Places API
and generate pitch decks for TOO GOOD TO GO sales representatives.
"""

import io
import json
import logging
from contextlib import redirect_stdout

import streamlit as st

from main import create_pitch_deck, print_pitch_deck, search_places

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
        ["🔍 Restaurant Search", "📊 Pitch Deck Generator"],
        index=0,
    )

    if page == "🔍 Restaurant Search":
        restaurant_search_page()
    elif page == "📊 Pitch Deck Generator":
        pitch_deck_page()


def restaurant_search_page():
    st.markdown(
        """
    <div class="section-header">
        <h2>🔍 Restaurant Search</h2>
        <p>Search for restaurants and stores in any location using Google Places API</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Input form
    with st.form("search_form"):
        st.subheader("Enter Search Query")
        query = st.text_input(
            "Search for restaurants:",
            placeholder="e.g., Chinese restaurants in Taufkirchen, Bavaria, Germany",
            help="Enter a search query like 'Italian restaurants in Munich' or 'sushi bars in Berlin'",
        )

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            st.info("The following fields are automatically included in the search:")
            st.write("• Restaurant name and address")
            st.write("• Contact information (phone, website)")
            st.write("• Ratings and reviews")
            st.write("• Opening hours and services")
            st.write("• Cuisine type and features")

        submitted = st.form_submit_button(
            "🔍 Search Restaurants", use_container_width=True
        )

    if submitted and query:
        search_restaurants(query)


def search_restaurants(query):
    # Define comprehensive fields for restaurant search
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
        "places.currentOpeningHours",
        "places.regularOpeningHours",
        "places.delivery",
        "places.dineIn",
        "places.takeout",
        "places.outdoorSeating",
        "places.servesBreakfast",
        "places.servesLunch",
        "places.servesDinner",
        "places.servesVegetarianFood",
        "places.editorialSummary",
        "places.priceLevel",
    ]

    with st.spinner("🔍 Searching for restaurants..."):
        try:
            results = search_places(query, fields_to_extract)

            if isinstance(results, dict) and "error" in results:
                st.error(f"❌ Error: {results['error']}")
                return

            if not results:
                st.warning("No restaurants found for your search query.")
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

            st.info(f"Found {len(results)} restaurant(s) matching your search.")

            display_all_restaurant_results(results)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")


def display_all_restaurant_results(restaurants):
    """Display all restaurant results with basic information in a list format"""
    for i, restaurant in enumerate(restaurants, 1):
        st.markdown('<div class="result-container">', unsafe_allow_html=True)

        # Restaurant name and basic info
        name = restaurant.get("displayName", {}).get("text", "Unknown Restaurant")
        address = restaurant.get("formattedAddress", "Address not available")

        col1, col2, col3 = st.columns([3, 2, 1])

        with col1:
            st.markdown(f"### {i}. 🏪 {name}")
            st.markdown(f"📍 {address}")

            # Cuisine type
            if restaurant.get("types"):
                types = restaurant["types"]
                cuisine_types = [
                    t.replace("_", " ").title()
                    for t in types
                    if "restaurant" in t.lower() or "food" in t.lower()
                ]
                if cuisine_types:
                    st.markdown(f"🍽️ **Type:** {', '.join(cuisine_types[:2])}")

        with col2:
            # Rating and reviews
            if restaurant.get("rating"):
                rating = restaurant["rating"]
                count = restaurant.get("userRatingCount", 0)
                st.metric("⭐ Rating", f"{rating}/5")
                st.caption(f"{count} reviews")

            # Business status
            if restaurant.get("businessStatus"):
                status = restaurant["businessStatus"].replace("_", " ").title()
                if status == "Operational":
                    st.success("✅ Open")
                else:
                    st.warning(f"⚠️ {status}")

        with col3:
            # Contact info
            if restaurant.get("internationalPhoneNumber"):
                st.markdown(f"📞 {restaurant['internationalPhoneNumber']}")

            if restaurant.get("websiteUri"):
                st.markdown(f"[🌐 Website]({restaurant['websiteUri']})")

            # Price level
            if restaurant.get("priceLevel"):
                try:
                    price_num = int(restaurant["priceLevel"])
                    price_level = "💰" * price_num
                    st.markdown(f"**Price:** {price_level}")
                except (ValueError, TypeError):
                    st.markdown(f"**Price:** {restaurant['priceLevel']}")

        # Quick features summary
        features = []
        if restaurant.get("delivery"):
            features.append("🚚 Delivery")
        if restaurant.get("dineIn"):
            features.append("🍽️ Dine-in")
        if restaurant.get("takeout"):
            features.append("🥡 Takeout")

        if features:
            st.markdown(f"**Services:** {' • '.join(features)}")

        # Show more details in expander
        with st.expander(f"📋 More details about {name}"):
            display_detailed_restaurant_info(restaurant)

        st.markdown("</div>", unsafe_allow_html=True)


def display_detailed_restaurant_info(restaurant):
    """Display detailed information about a single restaurant"""
    col1, col2 = st.columns(2)

    with col1:
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

        # Additional features
        features = []
        if restaurant.get("outdoorSeating"):
            features.append("🌤️ Outdoor Seating")
        if restaurant.get("goodForChildren"):
            features.append("👶 Kid-friendly")
        if restaurant.get("goodForGroups"):
            features.append("👥 Group-friendly")

        if features:
            st.markdown("**Features:** " + " • ".join(features))

    with col2:
        # Opening hours
        if restaurant.get("currentOpeningHours") or restaurant.get(
            "regularOpeningHours"
        ):
            st.markdown("**Opening Hours:**")
            hours = restaurant.get("currentOpeningHours") or restaurant.get(
                "regularOpeningHours"
            )
            if isinstance(hours, dict) and "weekdayDescriptions" in hours:
                for day_info in hours["weekdayDescriptions"]:
                    clean_day = " ".join(day_info.split())  # Clean up whitespace
                    st.caption(clean_day)

        # Editorial summary
        if restaurant.get("editorialSummary"):
            st.markdown("**Description:**")
            st.caption(restaurant["editorialSummary"].get("text", ""))


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
        st.subheader("Restaurant Information")
        query = st.text_input(
            "Enter restaurant query:",
            placeholder="e.g., Help me prepare a pitch deck for Pak Choi in Taufkirchen, Bavaria, Germany",
            help="Describe the restaurant you want to create a pitch deck for",
        )

        st.info(
            "💡 The pitch deck will include contact information, decision maker profile, key statistics, and tailored pitch strategies."
        )

        submitted = st.form_submit_button(
            "🚀 Generate Pitch Deck", use_container_width=True
        )

    if submitted and query:
        generate_pitch_deck(query)


def format_opening_hours(weekday_descriptions):
    """Format opening hours from weekdayDescriptions into a compact, readable format"""
    if not weekday_descriptions:
        return "Hours not available"

    # Simply join all descriptions with proper line breaks
    formatted_lines = []
    for day_info in weekday_descriptions:
        # Clean up extra whitespace and normalize
        clean_info = " ".join(day_info.split())
        if clean_info:
            formatted_lines.append(clean_info)

    return "\n".join(formatted_lines)


def format_opening_hours_string(hours_string):
    """Format opening hours from a string format into a compact, readable format"""
    if not hours_string:
        return "Hours not available"

    # Replace semicolons with line breaks and clean up
    clean_string = hours_string.replace(";", "\n")

    # Split into lines and clean each one
    lines = clean_string.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line:
            # Clean up extra whitespace
            line = " ".join(line.split())
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


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

    # Lead assessment info
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🔥 Lead Temperature", pitch_deck.get("lead_temperature", "unknown").title()
        )
    with col2:
        if pitch_deck.get("best_contact_time"):
            st.info(f"**Best Contact Time:** {pitch_deck['best_contact_time']}")

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
        st.metric(
            "🌱 Sustainability", stats.get("sustainability_signal", "unknown").title()
        )

    with col3:
        st.metric("💻 Digital Ready", stats.get("digital_readiness", "unknown").title())

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

    if strategy.get("social_proof"):
        st.write("**Social Proof Examples:**")
        for i, proof in enumerate(strategy["social_proof"], 1):
            st.write(f"{i}. {proof}")

    if strategy.get("urgency_closing"):
        st.write("**Closing Statement:**")
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
