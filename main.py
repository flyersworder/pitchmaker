# flake8: noqa: E501
"""
This script creates a pitch deck for a TOO GOOD TO GO sales representative based on a food business query.
It uses the Google Places API to extract food business information and enriches it with Google Search results.
The output is structured using Pydantic models for better organization and validation.
"""

import argparse
import json
import logging
import os
import sys
from enum import Enum
from typing import List, Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# Configure logging
def setup_logging(log_level=logging.INFO):
    """Set up logging configuration"""
    logger = logging.getLogger("pitchmaker")
    logger.setLevel(log_level)

    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add formatter to console handler
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logging()

# Load environment variables from .env file
load_dotenv()


# Define Pydantic models for structured output
class SustainabilityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class DigitalReadiness(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class LeadTemperature(str, Enum):
    COLD = "cold"
    WARM = "warm"
    HOT = "hot"


class ContactInfo(BaseModel):
    name: str
    address: str
    phone: Optional[str] = Field(
        default=None, description="Phone number of the food business"
    )
    email: Optional[str] = Field(
        default=None, description="Email address of the food business"
    )
    website: Optional[str] = Field(
        default=None, description="Website URL of the food business"
    )
    owner: Optional[str] = Field(
        default=None, description="Name of the food business owner or manager"
    )
    founded: Optional[str] = Field(
        default=None, description="When the food business was founded"
    )
    food_offerings: Optional[List[str]] = Field(
        default=None,
        description="Types of food offered (cuisine types, product categories, specialties)",
    )
    opening_hours: Optional[str] = Field(
        default=None, description="Opening hours of the food business"
    )


class DecisionMakerProfile(BaseModel):
    summary: str = Field(
        description="A brief summary of the decision maker or the food business management style"
    )
    pain_points: List[str] = Field(
        description="Potential pain points the decision maker might have regarding food waste"
    )
    values: List[str] = Field(
        description="Values that might be important to the decision maker"
    )
    communication_style: Optional[str] = Field(
        default=None, description="Recommended communication style for the pitch"
    )


class KeyStats(BaseModel):
    user_rating: Optional[float] = Field(
        default=None, description="Average user rating (out of 5)"
    )
    user_rating_count: Optional[int] = Field(
        default=None, description="Number of user ratings"
    )
    sustainability_signal: SustainabilityLevel = Field(
        default=SustainabilityLevel.UNKNOWN,
        description="Estimated sustainability level based on available information",
    )
    sustainability_reasoning: Optional[str] = Field(
        default=None, description="Reasoning behind the sustainability level assessment"
    )
    digital_readiness: DigitalReadiness = Field(
        default=DigitalReadiness.UNKNOWN,
        description="Estimated digital readiness based on online presence",
    )
    digital_readiness_reasoning: Optional[str] = Field(
        default=None, description="Reasoning behind the digital readiness assessment"
    )
    estimated_food_waste: Optional[str] = Field(
        default=None, description="Estimated daily/weekly food waste"
    )
    estimated_revenue_potential: Optional[str] = Field(
        default=None, description="Estimated revenue potential with Too Good To Go"
    )


class PersuasionTechnique(BaseModel):
    technique_name: str = Field(
        description="Name of the behavioral science technique (e.g., social proof, scarcity, etc.)"
    )
    description: str = Field(
        description="Brief description of how this technique works"
    )
    application: str = Field(
        description="Specific application of this technique for this food business"
    )
    effectiveness_reason: str = Field(
        description="Why this technique would be particularly effective for this food business"
    )
    pitch_script: str = Field(
        description="Ready-to-use script for sales representatives to deliver this persuasion technique in a natural, conversational way"
    )


class Objection(BaseModel):
    objection: str = Field(description="Common objection a food business might have")
    response: str = Field(description="How to address the objection")


class PitchStrategy(BaseModel):
    opening_hook: str = Field(
        description="Attention-grabbing opening line for the pitch"
    )
    value_proposition: str = Field(
        description="Clear value proposition tailored to this food business"
    )
    persuasion_techniques: List[PersuasionTechnique] = Field(
        description="Most effective behavioral science techniques for this specific food business",
        default_factory=list,
    )
    objection_handling: List[Objection] = Field(
        description="Common objections and how to address them"
    )
    urgency_closing: str = Field(
        description="Closing statement that creates urgency to sign up"
    )


class PitchDeck(BaseModel):
    contact_info: ContactInfo
    decision_maker_profile: DecisionMakerProfile
    key_stats: KeyStats
    pitch_strategy: PitchStrategy
    additional_notes: Optional[List[str]] = Field(
        default=None,
        description="Any additional insights or recommendations for the sales rep",
    )
    recommended_approach: str = Field(
        description="Overall recommended approach for pitching to this food business"
    )
    lead_temperature: LeadTemperature = Field(
        description="Assessment of how likely the food business is to convert (cold, warm, hot)"
    )
    lead_temperature_reasoning: Optional[str] = Field(
        default=None, description="Reasoning behind the lead temperature assessment"
    )
    best_contact_time: str = Field(
        description="Best time to contact the food business based on their business hours and type of establishment"
    )


def search_places(text_query: str, fields: list[str]) -> dict:
    """
    Searches for places using the Google Places API (Text Search).

    Args:
        text_query: The text string to search for (e.g., "restaurant in Sydney").
        fields: A list of fields to return in the response.
                Example: ['places.displayName', 'places.formattedAddress']

    Returns:
        A dictionary containing the API response or error information.
    """
    # Get the API key from environment variables
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error("GOOGLE_MAPS_API_KEY environment variable not set.")
        return {"error": "API key not found"}

    # Construct the request URL
    base_url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join(fields),
    }

    # Prepare the request payload
    payload = {"textQuery": text_query}

    logger.debug(f"Searching for places with query: {text_query}")
    logger.debug(f"Requesting fields: {', '.join(fields)}")

    try:
        # Make the API request
        response = requests.post(base_url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the response
        data = response.json()

        # Check if places were found
        if "places" not in data or not data["places"]:
            logger.warning(f"No places found for query: {text_query}")
            return {"error": "No places found", "query": text_query}

        logger.info(
            f"Found {len(data.get('places', []))} places for query: {text_query}"
        )
        # Return all places found
        return data["places"]

    except requests.exceptions.RequestException as e:
        logger.error(f"Error making Places API request: {e}")
        return {"error": str(e)}
    except ValueError as e:
        logger.error(f"Error parsing Places API response: {e}")
        return {"error": str(e)}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Exception: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return {"error": str(e)}


def create_pitch_deck(query):
    """Create a pitch deck for TOO GOOD TO GO sales rep based on a food business query."""
    # Initialize the Google Gemini API client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        return {"error": "API key not found"}

    client = genai.Client(api_key=api_key)

    logger.info(f"Processing user query: {query}")
    logger.info("Step 1: Extracting food business information using Places API...")

    # Step 1: Extract the food business name and location from the query
    extraction_prompt = f"""Extract ONLY the food business name and its location from this query: '{query}'.
    Format the response exactly as: 'food_business_name in location'.
    For example: 'Pak Choi in Taufkirchen, Bavaria, Germany'
    Do not include any other text or explanation."""

    extraction_response = client.models.generate_content(
        model="gemini-2.5-flash", contents=extraction_prompt
    )

    search_query = extraction_response.text.strip()
    logger.info(f"Extracted search query: {search_query}")

    # Define the fields we want to retrieve from the Places API
    # Using a more focused set of essential fields
    fields_to_extract = [
        # Basic information (Text Search Pro SKU)
        "places.displayName",
        "places.formattedAddress",
        "places.businessStatus",
        "places.types",
        "places.primaryType",
        "places.photos",
        "places.googleMapsUri",
        "places.shortFormattedAddress",
        # Additional details (Text Search Enterprise SKU)
        "places.priceLevel",
        "places.rating",
        "places.userRatingCount",
        "places.websiteUri",
        "places.internationalPhoneNumber",
        "places.currentOpeningHours",
        "places.regularOpeningHours",
        # Atmosphere and offerings (Enterprise + Atmosphere SKU)
        "places.delivery",
        "places.dineIn",
        "places.takeout",
        "places.outdoorSeating",
        "places.reservable",
        "places.goodForChildren",
        "places.goodForGroups",
        "places.servesBreakfast",
        "places.servesLunch",
        "places.servesDinner",
        "places.servesVegetarianFood",
        "places.editorialSummary",
        "places.reviews",
        "places.reviewSummary",
    ]

    # Directly call the search_places function
    logger.info(f"Searching for: {search_query}")
    places_data = search_places(search_query, fields_to_extract)

    # Check if we found any places or if there was an error
    if not places_data:
        logger.warning("No data returned from search_places tool.")
        return f"Could not find information about a food business matching '{search_query}'."

    if "error" in places_data:
        logger.error(f"Error in search_places: {places_data['error']}")
        # If we got an error, let's try a more generic search
        logger.info("Trying a more generic search...")
        # Extract just the food business name without location
        food_business_name_only = search_query.split(" in ")[0].strip()
        logger.info(
            f"Searching for just the food business name: {food_business_name_only}"
        )
        places_data = search_places(food_business_name_only, fields_to_extract)

        # Check again for errors or no results
        if not places_data or "error" in places_data:
            logger.error("Still could not find the food business.")
            return f"Could not find information about a food business named '{food_business_name_only}'."

    # At this point, places_data should be a list of food business objects
    # Select the first/most relevant result for pitch deck generation
    food_business = places_data[0] if isinstance(places_data, list) else places_data
    food_business_name = food_business.get("displayName", {}).get(
        "text", "Unknown Food Business"
    )
    food_business_address = food_business.get(
        "formattedAddress", "Address not available"
    )

    logger.info(f"Found food business: {food_business_name} at {food_business_address}")

    # Step 2: Use Google Search to enrich the information
    logger.info("\nStep 2: Enriching food business information with Google Search...\n")

    # Create Google Search tool
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    search_config = types.GenerateContentConfig(tools=[grounding_tool])

    # Make the request with Google Search tool
    enrichment_prompt = f"""
    ## CONTEXT: TOO GOOD TO GO BUSINESS MODEL
    TOO GOOD TO GO is a marketplace that connects consumers with food businesses (restaurants, cafes, bakeries, supermarkets, grocery stores, specialty food shops, etc.) that have surplus food. Businesses sell this surplus food at a reduced price through the TOO GOOD TO GO app, reducing food waste while generating additional revenue and reaching new customers. The business model benefits food businesses by:
    - Creating a new revenue stream from food that would otherwise be wasted
    - Attracting new customers who may become regulars
    - Enhancing sustainability credentials and brand reputation
    - Reducing waste disposal costs
    - Contributing to environmental sustainability goals

    ## TASK
    You are a specialized food business analyst for TOO GOOD TO GO, researching {food_business_name} located at {food_business_address} to identify partnership opportunities. Your goal is to gather comprehensive, factual information that will help our sales team craft a highly personalized and effective pitch. Conduct thorough research and provide ONLY factual, evidence-based information.

    ## SEARCH STRATEGY
    Begin with these search approaches, but feel free to pursue additional searches if you identify promising angles:
    1. Search for the business's official website, social media profiles, and online product/menu information
    2. Find recent reviews across multiple platforms (Google, Yelp, TripAdvisor, local blogs)
    3. Look for local news articles or business listings mentioning the business
    4. Research existing TOO GOOD TO GO partners in the same region or with similar offerings
    5. Identify information about the business's operations, management, and customer experiences

    IMPORTANT: Whenever possible, conduct searches in the local language of the business's location, as this typically yields richer and more detailed results. Local language searches often reveal information not available in English, including authentic customer reviews, local news coverage, and cultural context that can be valuable for crafting a personalized pitch.

    ## INFORMATION TO GATHER
    Focus on these categories, but add any additional relevant information you discover:

    ### FOOD WASTE POTENTIAL
    - Product/menu variety and complexity (more items = higher waste potential)
    - Daily specials, seasonal items, or prepared food offerings
    - Fresh items with short shelf life (e.g., produce, baked goods, prepared meals)
    - Portion sizes or package sizes mentioned in reviews
    - Any explicit mentions of food waste practices or sustainability initiatives
    - For retail: inventory management practices, markdown policies, or expiration date handling

    ### OPERATIONAL PATTERNS
    - Peak business hours vs. slow periods
    - Seasonal fluctuations in business
    - Recent changes: expansion, renovation, product/menu updates, management changes
    - Operational challenges mentioned in reviews or articles
    - For restaurants/cafes: Delivery/takeout vs. dine-in balance
    - For retail: Shopping patterns, busiest days, inventory turnover

    ### MANAGEMENT INSIGHTS
    - Owner/manager names and management style if available
    - Decision-making approach (traditional vs. innovative)
    - Responsiveness to customer feedback
    - Involvement in community or local events
    - Any mentions of business priorities or values

    ### COMPETITIVE LANDSCAPE
    - Similar food businesses in the area
    - Existing TOO GOOD TO GO partners in the same region or with similar offerings
    - Local competitors using food waste reduction services
    - Unique selling points compared to competitors
    - Position in the local market (upscale, mid-range, budget)
    - For retail: Competing stores, product overlap, pricing strategy

    ### CUSTOMER SENTIMENT
    - Overall satisfaction with product/food quality and value
    - Specific praise or complaints about portions, packaging, or quality
    - Mentions of price-to-value ratio
    - Customer demographic insights
    - For retail: Customer feedback on product selection, freshness, and shopping experience

    ## OUTPUT FORMAT
    Provide factual, concise information in bullet points under each category. For each insight, include a brief note on how it might be relevant to a TOO GOOD TO GO partnership opportunity.

    If you discover any particularly compelling insights that don't fit neatly into the categories above, include them in an "ADDITIONAL INSIGHTS" section.

    If information is not available for certain categories, explicitly state this rather than making assumptions. Remember that high-quality, specific information is more valuable than general observations.

    ## FINAL ASSESSMENT
    End your analysis with a brief (2-3 sentence) assessment of this food business's potential fit with TOO GOOD TO GO, highlighting the most promising partnership angles based on your research.
    """

    search_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=enrichment_prompt,
        config=search_config,
    )

    logger.info("Google Search enrichment completed.")
    logger.info(f"Google Search response: {search_response.text}")

    # Step 3: Generate the pitch deck content
    logger.info("\nStep 3: Creating TOO GOOD TO GO pitch deck content...\n")

    # Format food business data for the prompt, handling nested structures and missing fields
    food_business_data = {
        "name": food_business_name,
        "address": food_business_address,
        "location": (
            search_query.split(" in ", 1)[1] if " in " in search_query else "Unknown"
        ),
    }

    # Add basic fields that might be directly in the food business object
    for field, display_name in [
        ("priceLevel", "price_level"),
        ("websiteUri", "website"),
        ("internationalPhoneNumber", "phone"),
        ("rating", "rating"),
        ("userRatingCount", "user_rating_count"),
        ("businessStatus", "business_status"),
        ("googleMapsUri", "google_maps_url"),
        ("shortFormattedAddress", "short_address"),
    ]:
        if field in food_business:
            food_business_data[display_name] = food_business[field]

    # Add types if available
    if "types" in food_business:
        food_business_data["types"] = food_business["types"]
    elif "primaryType" in food_business:
        food_business_data["primary_type"] = food_business["primaryType"]

    # Add photos if available
    if "photos" in food_business and food_business["photos"]:
        food_business_data["has_photos"] = True
        food_business_data["photo_count"] = len(food_business["photos"])

    # Add opening hours
    if "currentOpeningHours" in food_business:
        food_business_data["opening_hours"] = food_business["currentOpeningHours"]
    elif "regularOpeningHours" in food_business:
        food_business_data["opening_hours"] = food_business["regularOpeningHours"]

    # Add food business features and services
    for feature, display_name in [
        ("delivery", "offers_delivery"),
        ("dineIn", "offers_dine_in"),
        ("takeout", "offers_takeout"),
        ("outdoorSeating", "has_outdoor_seating"),
        ("reservable", "is_reservable"),
        ("goodForChildren", "good_for_children"),
        ("goodForGroups", "good_for_groups"),
    ]:
        if feature in food_business:
            food_business_data[display_name] = food_business[feature]

    # Add meal service information
    for service, display_name in [
        ("servesBreakfast", "serves_breakfast"),
        ("servesLunch", "serves_lunch"),
        ("servesDinner", "serves_dinner"),
        ("servesVegetarianFood", "serves_vegetarian"),
    ]:
        if service in food_business:
            food_business_data[display_name] = food_business[service]

    # Add editorial and review information
    if "editorialSummary" in food_business:
        food_business_data["editorial_summary"] = food_business["editorialSummary"]
    if "reviewSummary" in food_business:
        food_business_data["review_summary"] = food_business["reviewSummary"]
    if "reviews" in food_business:
        food_business_data["reviews"] = food_business["reviews"]

    # Log the complete food business data for debugging
    logger.debug("Food business data from Places API:")
    logger.debug(json.dumps(food_business_data, indent=2))

    # Create the structured pitch deck prompt
    pitch_deck_prompt = f"""
    ## ROLE AND OBJECTIVE
    You are an expert pitch deck creator for TOO GOOD TO GO, specializing in crafting evidence-based, persuasive sales materials. Your goal is to create a highly effective, personalized pitch deck for {food_business_name} that will maximize conversion likelihood during an initial phone call pitch. While this business may be referred to as a restaurant in some places, apply the same approach for any food business type (bakery, café, supermarket, grocery store, specialty food shop, etc.).

    ## AVAILABLE DATA
    FOOD BUSINESS INFORMATION FROM PLACES API:
    ```json
    {json.dumps(food_business_data, indent=2)}
    ```

    ADDITIONAL INFORMATION FROM GOOGLE SEARCH:
    ```
    {search_response.text}
    ```

    ## INSTRUCTIONS
    1. Base all content on verifiable evidence from the provided data. If information is missing, indicate this clearly rather than making assumptions.
    2. Create a structured pitch deck with the following sections:

    ## REQUIRED SECTIONS
    1. **Contact Information**: Complete details about {food_business_name} (name, address, phone, website, cuisine type, etc.)

    2. **Decision Maker Profile**: Evidence-based assessment of management style, priorities, and pain points. Include:
       - Likely decision-making approach based on business operations
       - Key pain points related to food waste or inventory management
       - Core values that would align with TOO GOOD TO GO's mission

     3. **Key Statistics**: Data-driven assessment including:
        - Rating and review summary
        - Sustainability signal (high/medium/low) with detailed reasoning for the assessment
        - Digital readiness assessment (high/medium/low) with detailed reasoning for the assessment
        - Estimated food waste potential and revenue opportunity

     4. **Phone Call Pitch Strategy**: Scientifically-grounded approach using behavioral science principles:
        - Opening hook that quickly engages and addresses specific pain points (10-15 seconds)
        - Value proposition tailored to {food_business_name}'s specific situation with clear vocal emphasis points
        - 3-4 most effective persuasion techniques for this specific food business, including:
          * Detailed application of each technique for this food business
          * Why each technique would be particularly effective in this case
          * For social proof, prioritize examples that most closely resemble this food business (similar offerings, size, price point, or location). When multiple examples are available, select the ones with the highest similarity to maximize relevance and impact
          * A ready-to-use pitch script (2-4 sentences) for each technique that sales representatives can use verbatim in conversation
        - Responses to 2-3 likely objections that might arise during the call
        - Conversation transitions and questions to maintain engagement
        - Urgency-based closing that creates momentum toward scheduling a follow-up meeting

     5. **Behavioral Science Insights**: Specific tactical recommendations based on:
        - Which behavioral principles (social proof, scarcity, etc.) will be most effective for this specific food business
        - How to frame the TOO GOOD TO GO value proposition to align with the food business's values
        - Psychological triggers that would resonate with this specific decision maker

    6. **Lead Assessment**:
       - Temperature rating (cold/warm/hot) with detailed reasoning for the assessment
       - Optimal contact time based on business operations and decision maker availability

    ## TONE AND STYLE
    Make the pitch deck concise, persuasive, and specifically tailored to {food_business_name} for an effective phone call pitch.
    Use concrete details from the data to personalize every aspect of the pitch.
    Focus on conversational language, clear talking points, and actionable insights that will help the sales representative engage the listener and move toward scheduling a follow-up meeting.
    Remember that this is for a verbal phone conversation, so include natural transitions, pauses, and questions that would work well in spoken dialogue.

    ## LANGUAGE REQUIREMENTS
    IMPORTANT: Create the entire pitch deck in English only, regardless of the food business's location or language. Do not include any content in other languages, even for greetings or cultural references. The pitch should be fully accessible to English-speaking sales representatives.
    """

    # Generate the structured pitch deck content
    pitch_deck_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=pitch_deck_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": PitchDeck,
        },
    )

    logger.info("Generating structured pitch deck for TOO GOOD TO GO sales rep...")

    # Parse the structured output
    try:
        # Use the parsed response directly
        if hasattr(pitch_deck_response, "parsed"):
            # Get the parsed Pydantic object with proper enum handling
            pitch_deck_model = pitch_deck_response.parsed

            # Convert to dictionary with enum values as strings
            pitch_deck = pitch_deck_model.model_dump()

            # Extract the enum values properly
            if isinstance(
                pitch_deck_model.key_stats.sustainability_signal, SustainabilityLevel
            ):
                pitch_deck["key_stats"][
                    "sustainability_signal"
                ] = pitch_deck_model.key_stats.sustainability_signal.value

            if isinstance(
                pitch_deck_model.key_stats.digital_readiness, DigitalReadiness
            ):
                pitch_deck["key_stats"][
                    "digital_readiness"
                ] = pitch_deck_model.key_stats.digital_readiness.value

            if isinstance(pitch_deck_model.lead_temperature, LeadTemperature):
                pitch_deck["lead_temperature"] = pitch_deck_model.lead_temperature.value
        else:
            # Fallback to manual JSON parsing if parsed attribute is not available
            pitch_deck = json.loads(pitch_deck_response.text)

            # Process enum values before returning
            # Helper function to clean enum values
            def extract_enum_value(value):
                if not isinstance(value, str):
                    value = str(value)
                # Check for enum class prefixes and extract the value part
                if "." in value:
                    parts = value.split(".")
                    if len(parts) == 2:
                        return parts[1].lower()
                return value

            # Handle sustainability signal
            if (
                "key_stats" in pitch_deck
                and "sustainability_signal" in pitch_deck["key_stats"]
            ):
                pitch_deck["key_stats"]["sustainability_signal"] = extract_enum_value(
                    pitch_deck["key_stats"]["sustainability_signal"]
                )

            # Handle digital readiness
            if (
                "key_stats" in pitch_deck
                and "digital_readiness" in pitch_deck["key_stats"]
            ):
                pitch_deck["key_stats"]["digital_readiness"] = extract_enum_value(
                    pitch_deck["key_stats"]["digital_readiness"]
                )

            # Handle lead temperature
            if "lead_temperature" in pitch_deck:
                pitch_deck["lead_temperature"] = extract_enum_value(
                    pitch_deck["lead_temperature"]
                )

        logger.info("Successfully parsed pitch deck response")
        return pitch_deck
    except Exception as e:
        logger.error(f"Error parsing structured output: {e}")
        return {"error": str(e), "raw_response": pitch_deck_response.text}


def print_pitch_deck(pitch_deck):
    """Print a formatted version of the pitch deck."""
    print("\nFormatted Pitch Deck:")

    print("\n=== CONTACT INFORMATION ===\n")
    print(f"Food Business: {pitch_deck['contact_info']['name']}")
    print(f"Address: {pitch_deck['contact_info']['address']}")
    if "phone" in pitch_deck["contact_info"] and pitch_deck["contact_info"]["phone"]:
        print(f"Phone: {pitch_deck['contact_info']['phone']}")
    if (
        "website" in pitch_deck["contact_info"]
        and pitch_deck["contact_info"]["website"]
    ):
        print(f"Website: {pitch_deck['contact_info']['website']}")
    if (
        "cuisine_type" in pitch_deck["contact_info"]
        and pitch_deck["contact_info"]["cuisine_type"]
    ):
        print(f"Cuisine: {', '.join(pitch_deck['contact_info']['cuisine_type'])}")

    print("\n=== DECISION MAKER PROFILE ===\n")
    print(pitch_deck["decision_maker_profile"]["summary"])
    print("\nPain Points:")
    for point in pitch_deck["decision_maker_profile"]["pain_points"]:
        print(f"- {point}")
    print("\nValues:")
    for value in pitch_deck["decision_maker_profile"]["values"]:
        print(f"- {value}")

    print("\n=== KEY STATS ===\n")
    if (
        "user_rating" in pitch_deck["key_stats"]
        and pitch_deck["key_stats"]["user_rating"]
    ):
        print(
            f"Rating: {pitch_deck['key_stats']['user_rating']}/5 ({pitch_deck['key_stats'].get('user_rating_count', 'N/A')} reviews)"
        )

    # Print sustainability signal and digital readiness with reasoning
    print(f"Sustainability Signal: {pitch_deck['key_stats']['sustainability_signal']}")
    if (
        "sustainability_reasoning" in pitch_deck["key_stats"]
        and pitch_deck["key_stats"]["sustainability_reasoning"]
    ):
        print(f"Reasoning: {pitch_deck['key_stats']['sustainability_reasoning']}")

    print(f"Digital Readiness: {pitch_deck['key_stats']['digital_readiness']}")
    if (
        "digital_readiness_reasoning" in pitch_deck["key_stats"]
        and pitch_deck["key_stats"]["digital_readiness_reasoning"]
    ):
        print(f"Reasoning: {pitch_deck['key_stats']['digital_readiness_reasoning']}")

    if (
        "estimated_food_waste" in pitch_deck["key_stats"]
        and pitch_deck["key_stats"]["estimated_food_waste"]
    ):
        print(f"Est. Food Waste: {pitch_deck['key_stats']['estimated_food_waste']}")
    if (
        "estimated_revenue_potential" in pitch_deck["key_stats"]
        and pitch_deck["key_stats"]["estimated_revenue_potential"]
    ):
        print(
            f"Est. Revenue Potential: {pitch_deck['key_stats']['estimated_revenue_potential']}"
        )

    print("\n=== PITCH STRATEGY ===\n")
    print(f"Opening Hook: {pitch_deck['pitch_strategy']['opening_hook']}")
    print(f"\nValue Proposition: {pitch_deck['pitch_strategy']['value_proposition']}")

    print("\nPersuasion Techniques:")
    for technique in pitch_deck["pitch_strategy"]["persuasion_techniques"]:
        print(f"\n- {technique['technique_name']}")
        print(f"  Description: {technique['description']}")
        print(f"  Application: {technique['application']}")
        print(f"  Why effective: {technique['effectiveness_reason']}")
        print(f"\n  PITCH SCRIPT:\n  \"{technique['pitch_script']}\"")

    print("\nObjection Handling:")
    for item in pitch_deck["pitch_strategy"]["objection_handling"]:
        print(f"- {item['objection']}: {item['response']}")

    print(f"\nClosing: {pitch_deck['pitch_strategy']['urgency_closing']}")

    if "additional_notes" in pitch_deck and pitch_deck["additional_notes"]:
        print("\n=== ADDITIONAL NOTES ===\n")
        for note in pitch_deck["additional_notes"]:
            print(f"- {note}")

    print("\n=== RECOMMENDED APPROACH ===\n")
    print(pitch_deck["recommended_approach"])

    print("\n=== LEAD ASSESSMENT ===\n")

    # Print lead temperature with reasoning
    print(f"Lead Temperature: {pitch_deck['lead_temperature']}")
    if (
        "lead_temperature_reasoning" in pitch_deck
        and pitch_deck["lead_temperature_reasoning"]
    ):
        print(f"Reasoning: {pitch_deck['lead_temperature_reasoning']}")

    print(f"Best Contact Time: {pitch_deck['best_contact_time']}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TOO GOOD TO GO Pitch Deck Creator")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="Help me prepare a pitch deck for a restaurant called Pak Choi in Taufkirchen in Bavaria, Germany",
        help="Food business query to search for",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level)",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set logging level based on arguments
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(getattr(logging, args.log_level))

    logger.info("===== TOO GOOD TO GO Pitch Deck Creator =====")

    # Generate the pitch deck
    pitch_deck = create_pitch_deck(args.query)

    # Log the raw JSON response at debug level
    logger.debug("Raw JSON Response:")
    logger.debug(json.dumps(pitch_deck, indent=2))

    # Print the formatted pitch deck
    print_pitch_deck(pitch_deck)
