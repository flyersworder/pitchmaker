# flake8: noqa: E501
"""
This script creates a pitch deck for a TOO GOOD TO GO sales representative based on a restaurant query.
It uses the Google Places API to extract restaurant information and enriches it with Google Search results.
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
        default=None, description="Phone number of the restaurant"
    )
    email: Optional[str] = Field(
        default=None, description="Email address of the restaurant"
    )
    website: Optional[str] = Field(
        default=None, description="Website URL of the restaurant"
    )
    owner: Optional[str] = Field(
        default=None, description="Name of the restaurant owner or manager"
    )
    founded: Optional[str] = Field(
        default=None, description="When the restaurant was founded"
    )
    cuisine_type: Optional[List[str]] = Field(
        default=None, description="Types of cuisine served"
    )
    opening_hours: Optional[str] = Field(
        default=None, description="Opening hours of the restaurant"
    )


class DecisionMakerProfile(BaseModel):
    summary: str = Field(
        description="A brief summary of the decision maker or the restaurant management style"
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
    digital_readiness: DigitalReadiness = Field(
        default=DigitalReadiness.UNKNOWN,
        description="Estimated digital readiness based on online presence",
    )
    estimated_food_waste: Optional[str] = Field(
        default=None, description="Estimated daily/weekly food waste"
    )
    estimated_revenue_potential: Optional[str] = Field(
        default=None, description="Estimated revenue potential with Too Good To Go"
    )


class Objection(BaseModel):
    objection: str = Field(description="Common objection a restaurant might have")
    response: str = Field(description="How to address the objection")


class PitchStrategy(BaseModel):
    opening_hook: str = Field(
        description="Attention-grabbing opening line for the pitch"
    )
    value_proposition: str = Field(
        description="Clear value proposition tailored to this restaurant"
    )
    social_proof: List[str] = Field(
        description="Examples of similar restaurants that have succeeded with Too Good To Go"
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
        description="Any additional information that might help with the pitch",
    )
    recommended_approach: str = Field(
        description="Overall recommended approach for pitching to this restaurant"
    )
    lead_temperature: LeadTemperature = Field(
        description="Assessment of how likely the restaurant is to convert (cold, warm, hot)"
    )
    best_contact_time: str = Field(
        description="Best time to contact the restaurant based on their business hours and type of establishment"
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
        # Return the first place found
        return data["places"][0]

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
        print(f"Unexpected Error: {e}")
        return {"error": str(e)}


def create_pitch_deck(query):
    """Create a pitch deck for TOO GOOD TO GO sales rep based on a restaurant query."""
    # Initialize the Google Gemini API client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        return {"error": "API key not found"}

    client = genai.Client(api_key=api_key)

    logger.info(f"Processing user query: {query}")
    logger.info("Step 1: Extracting restaurant information using Places API...")

    # Step 1: Extract the restaurant name and location from the query
    extraction_prompt = f"""Extract ONLY the restaurant name and its location from this query: '{query}'.
    Format the response exactly as: 'restaurant_name in location'.
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
        return (
            f"Could not find information about a restaurant matching '{search_query}'."
        )

    if "error" in places_data:
        logger.error(f"Error in search_places: {places_data['error']}")
        # If we got an error, let's try a more generic search
        logger.info("Trying a more generic search...")
        # Extract just the restaurant name without location
        restaurant_name_only = search_query.split(" in ")[0].strip()
        logger.info(f"Searching for just the restaurant name: {restaurant_name_only}")
        places_data = search_places(restaurant_name_only, fields_to_extract)

        # Check again for errors or no results
        if not places_data or "error" in places_data:
            logger.error("Still could not find the restaurant.")
            return f"Could not find information about a restaurant named '{restaurant_name_only}'."

    # At this point, places_data should be a single restaurant object
    # (the first/most relevant result from the search)
    restaurant = places_data
    restaurant_name = restaurant.get("displayName", {}).get(
        "text", "Unknown Restaurant"
    )
    restaurant_address = restaurant.get("formattedAddress", "Address not available")

    print(f"Found restaurant: {restaurant_name} at {restaurant_address}")

    # Step 2: Use Google Search to enrich the information
    print("\nStep 2: Enriching restaurant information with Google Search...\n")

    # Create Google Search tool
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    search_config = types.GenerateContentConfig(tools=[grounding_tool])

    # Make the request with Google Search tool
    enrichment_prompt = f"""Find detailed information about the restaurant '{restaurant_name}' located at {restaurant_address}' specifically focused on aspects relevant for a TOO GOOD TO GO partnership.
    Focus on:
    1. Food waste management practices (if any)
    2. Sustainability initiatives or eco-friendly practices
    3. Menu variety and items that might be suitable for surplus food packages
    4. Operational challenges that might lead to food waste (e.g., large menu, daily specials, buffet offerings)
    5. Busy vs. slow periods that might affect food surplus
    6. Management style and decision-making approach
    7. Recent changes (expansion, renovation, menu changes) that might affect operations
    8. Customer sentiment about value and quality
    9. Competitors nearby who are already using food waste reduction services
    10. Any unique challenges in their business model related to inventory or food preparation
    11. Local community engagement or neighborhood reputation
    12. Any mentions of excess food, portions, or quantity in reviews
    """

    search_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=enrichment_prompt,
        config=search_config,
    )

    print("Google Search enrichment completed.")

    # Step 3: Generate the pitch deck content
    print("\nStep 3: Creating TOO GOOD TO GO pitch deck content...\n")

    # Format restaurant data for the prompt, handling nested structures and missing fields
    restaurant_data = {
        "name": restaurant_name,
        "address": restaurant_address,
        "location": (
            search_query.split(" in ", 1)[1] if " in " in search_query else "Unknown"
        ),
    }

    # Add basic fields that might be directly in the restaurant object
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
        if field in restaurant:
            restaurant_data[display_name] = restaurant[field]

    # Add types if available
    if "types" in restaurant:
        restaurant_data["types"] = restaurant["types"]
    elif "primaryType" in restaurant:
        restaurant_data["primary_type"] = restaurant["primaryType"]

    # Add photos if available
    if "photos" in restaurant and restaurant["photos"]:
        restaurant_data["has_photos"] = True
        restaurant_data["photo_count"] = len(restaurant["photos"])

    # Add opening hours
    if "currentOpeningHours" in restaurant:
        restaurant_data["opening_hours"] = restaurant["currentOpeningHours"]
    elif "regularOpeningHours" in restaurant:
        restaurant_data["opening_hours"] = restaurant["regularOpeningHours"]

    # Add restaurant features and services
    for feature, display_name in [
        ("delivery", "offers_delivery"),
        ("dineIn", "offers_dine_in"),
        ("takeout", "offers_takeout"),
        ("outdoorSeating", "has_outdoor_seating"),
        ("reservable", "is_reservable"),
        ("goodForChildren", "good_for_children"),
        ("goodForGroups", "good_for_groups"),
    ]:
        if feature in restaurant:
            restaurant_data[display_name] = restaurant[feature]

    # Add meal service information
    for service, display_name in [
        ("servesBreakfast", "serves_breakfast"),
        ("servesLunch", "serves_lunch"),
        ("servesDinner", "serves_dinner"),
        ("servesVegetarianFood", "serves_vegetarian"),
    ]:
        if service in restaurant:
            restaurant_data[display_name] = restaurant[service]

    # Add editorial and review information
    if "editorialSummary" in restaurant:
        restaurant_data["editorial_summary"] = restaurant["editorialSummary"]
    if "reviewSummary" in restaurant:
        restaurant_data["review_summary"] = restaurant["reviewSummary"]
    if "reviews" in restaurant:
        restaurant_data["reviews"] = restaurant["reviews"]

    # Log the complete restaurant data for debugging
    logger.debug("Restaurant data from Places API:")
    logger.debug(json.dumps(restaurant_data, indent=2))

    # Create the structured pitch deck prompt
    pitch_deck_prompt = f"""
    You are a professional pitch deck creator for TOO GOOD TO GO, a company that helps restaurants reduce food waste by selling surplus food at a discount.

    Create a structured pitch deck for a TOO GOOD TO GO sales representative to use when pitching to {restaurant_name}.

    RESTAURANT INFORMATION FROM PLACES API:
    {json.dumps(restaurant_data, indent=2)}

    ADDITIONAL INFORMATION FROM GOOGLE SEARCH:
    {search_response.text}

    Your response should include:

    1. Contact information about the restaurant (name, address, phone, website, etc.)
    2. A profile of the likely decision maker or management style
    3. Key statistics about the restaurant (ratings, sustainability signals, digital readiness)
    4. Pitch strategies (including opening hook, social proof examples, and urgency closing) that leverage effective behavior science theories (e.g., social proof, scarcity, reciprocity, loss aversion, authority) to maximize conversion likelihood
    5. Any additional information, insights, or context from behavioral science that would help the sales representative tailor their approach
    6. Lead temperature assessment (cold, warm, or hot) based on how likely the restaurant is to convert
    7. Best time to contact the restaurant based on their business hours and type of establishment

    Make the pitch deck concise, persuasive, and tailored specifically to {restaurant_name} based on the information provided.
    Use specific details about the restaurant to personalize the pitch.
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
    print(f"Restaurant: {pitch_deck['contact_info']['name']}")
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

    # Print sustainability signal and digital readiness
    print(f"Sustainability Signal: {pitch_deck['key_stats']['sustainability_signal']}")
    print(f"Digital Readiness: {pitch_deck['key_stats']['digital_readiness']}")

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

    print("\nSocial Proof Examples:")
    for proof in pitch_deck["pitch_strategy"]["social_proof"]:
        print(f"- {proof}")

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

    # Print lead temperature
    print(f"Lead Temperature: {pitch_deck['lead_temperature']}")

    print(f"Best Contact Time: {pitch_deck['best_contact_time']}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TOO GOOD TO GO Pitch Deck Creator")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default="Help me prepare a pitch deck for a restaurant called Pak Choi in Taufkirchen in Bavaria, Germany",
        help="Restaurant query to search for",
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
