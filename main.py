# flake8: noqa: E501
"""
This script creates a pitch deck for a TOO GOOD TO GO sales representative based on a restaurant query.
It uses the Google Places API to extract restaurant information and enriches it with Google Search results.
The output is structured using Pydantic models for better organization and validation.
"""

import json
import os
from enum import Enum
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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
    objection_handling: Dict[str, str] = Field(
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
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join(fields),
    }

    data = {"textQuery": text_query}

    print(f"API Request - URL: {url}")
    print(f"API Request - Headers: {headers}")
    print(f"API Request - Data: {data}")

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response content: {e.response.text}")
        return {"error": str(e), "status_code": e.response.status_code}
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
        return {"error": str(e)}
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error: {e}")
        return {"error": str(e)}
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return {"error": str(e)}


def create_pitch_deck(query):
    """Create a pitch deck for TOO GOOD TO GO sales rep based on a restaurant query."""
    client = genai.Client()

    print(f"\nUser query: {query}")
    print("Step 1: Extracting restaurant information using Places API...\n")

    # Step 1: Extract the restaurant name and location from the query
    extraction_prompt = f"""Extract ONLY the restaurant name and its location from this query: '{query}'.
    Format the response exactly as: 'restaurant_name in location'.
    For example: 'Pak Choi in Taufkirchen, Bavaria, Germany'
    Do not include any other text or explanation."""

    extraction_response = client.models.generate_content(
        model="gemini-2.5-flash", contents=extraction_prompt
    )

    search_query = extraction_response.text.strip()
    print(f"Extracted search query: {search_query}")

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
        # Additional details (Text Search Enterprise SKU)
        "places.priceLevel",
        "places.rating",
        "places.userRatingCount",
        "places.websiteUri",
        "places.internationalPhoneNumber",
    ]

    # Directly call the search_places function
    print(f"Searching for: {search_query}")
    places_data = search_places(search_query, fields_to_extract)

    # Check if we found any places or if there was an error
    if not places_data:
        print("No data returned from search_places tool.")
        return (
            f"Could not find information about a restaurant matching '{search_query}'."
        )

    if "error" in places_data:
        print(f"Error in search_places: {places_data['error']}")
        # If we got an error, let's try a more generic search
        print("Trying a more generic search...")
        # Extract just the restaurant name without location
        restaurant_name_only = search_query.split(" in ")[0].strip()
        print(f"Searching for just the restaurant name: {restaurant_name_only}")
        places_data = search_places(restaurant_name_only, fields_to_extract)

        # Check again for errors or no results
        if (
            not places_data
            or "error" in places_data
            or "places" not in places_data
            or not places_data["places"]
        ):
            print("Still could not find the restaurant.")
            return f"Could not find information about a restaurant named '{restaurant_name_only}'."

    if "places" not in places_data or not places_data["places"]:
        print("No restaurant found in the search results.")
        return (
            f"Could not find information about a restaurant matching '{search_query}'."
        )

    # Get the first (most relevant) restaurant
    restaurant = places_data["places"][0]
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
    enrichment_prompt = f"""Find detailed information about the restaurant '{restaurant_name}' located at {restaurant_address}.
    Focus on:
    1. Type of cuisine and specialties
    2. Customer reviews and ratings
    3. Sustainability practices (if any)
    4. Food waste management (if any)
    5. Popular dishes
    6. Opening hours
    7. Busy times and peak hours
    8. Average price range
    9. Target customer demographic
    10. Any unique selling points or special features
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

    # Add fields that might be directly in the restaurant object
    for field, display_name in [
        ("priceLevel", "price_level"),
        ("websiteUri", "website"),
        ("internationalPhoneNumber", "phone"),
        ("rating", "rating"),
        ("userRatingCount", "user_rating_count"),
        ("businessStatus", "business_status"),
    ]:
        if field in restaurant:
            restaurant_data[display_name] = restaurant[field]

    # Add types if available
    if "types" in restaurant:
        restaurant_data["types"] = restaurant["types"]
    elif "primaryType" in restaurant:
        restaurant_data["primary_type"] = restaurant["primaryType"]

    # Add any other available fields that might be useful
    if "photos" in restaurant and restaurant["photos"]:
        restaurant_data["has_photos"] = True
        restaurant_data["photo_count"] = len(restaurant["photos"])

    # Print the complete restaurant data for debugging
    print("\nRestaurant data from Places API:")
    print(json.dumps(restaurant_data, indent=2))

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
    4. Pitch strategies including opening hook, social proof examples, and urgency closing
    5. Any additional information that would help the sales representative
    6. Lead temperature assessment (cold, warm, or hot) based on how likely the restaurant is to convert
    7. Best time to contact the restaurant based on their business hours and type of establishment

    Make the pitch deck concise, persuasive, and tailored specifically to {restaurant_name} based on the information provided.
    Use specific details about the restaurant to personalize the pitch.
    """

    # Define the schema directly as a dictionary (compatible with Gemini API)
    schema = {
        "type": "object",
        "properties": {
            "contact_info": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "string"},
                    "phone": {"type": "string"},
                    "website": {"type": "string"},
                    "email": {"type": "string"},
                    "owner": {"type": "string"},
                    "founded": {"type": "string"},
                    "cuisine_type": {"type": "array", "items": {"type": "string"}},
                    "opening_hours": {"type": "string"},
                },
                "required": ["name", "address"],
            },
            "decision_maker_profile": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "pain_points": {"type": "array", "items": {"type": "string"}},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "communication_style": {"type": "string"},
                },
                "required": ["summary", "pain_points", "values"],
            },
            "key_stats": {
                "type": "object",
                "properties": {
                    "user_rating": {"type": "number"},
                    "user_rating_count": {"type": "integer"},
                    "sustainability_signal": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "unknown"],
                    },
                    "digital_readiness": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "unknown"],
                    },
                    "estimated_food_waste": {"type": "string"},
                    "estimated_revenue_potential": {"type": "string"},
                },
                "required": ["sustainability_signal", "digital_readiness"],
            },
            "pitch_strategy": {
                "type": "object",
                "properties": {
                    "opening_hook": {"type": "string"},
                    "value_proposition": {"type": "string"},
                    "social_proof": {"type": "array", "items": {"type": "string"}},
                    "objection_handling": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "objection": {"type": "string"},
                                "response": {"type": "string"},
                            },
                            "required": ["objection", "response"],
                        },
                    },
                    "urgency_closing": {"type": "string"},
                },
                "required": [
                    "opening_hook",
                    "value_proposition",
                    "social_proof",
                    "urgency_closing",
                ],
            },
            "additional_notes": {"type": "array", "items": {"type": "string"}},
            "recommended_approach": {"type": "string"},
            "lead_temperature": {"type": "string", "enum": ["cold", "warm", "hot"]},
            "best_contact_time": {"type": "string"},
        },
        "required": [
            "contact_info",
            "decision_maker_profile",
            "key_stats",
            "pitch_strategy",
            "recommended_approach",
            "lead_temperature",
            "best_contact_time",
        ],
    }

    # Generate the structured pitch deck content
    pitch_deck_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=pitch_deck_prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    print("\nStructured Pitch Deck for TOO GOOD TO GO Sales Rep:")

    # Parse the structured output
    try:
        # Get the parsed JSON object
        pitch_deck = json.loads(pitch_deck_response.text)
        return pitch_deck
    except Exception as e:
        print(f"Error parsing structured output: {e}")
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
    print(f"Lead Temperature: {pitch_deck['lead_temperature']}")
    print(f"Best Contact Time: {pitch_deck['best_contact_time']}")


if __name__ == "__main__":
    # Example usage
    print("\n===== TOO GOOD TO GO Pitch Deck Creator =====")
    user_query = "Help me prepare a pitch deck for a restaurant called Pak Choi in Taufkirchen in Bavaria, Germany"

    # Generate the pitch deck
    pitch_deck = create_pitch_deck(user_query)

    # Print the raw JSON response
    print("\nJSON Response:")
    print(json.dumps(pitch_deck, indent=2))

    # Print the formatted pitch deck
    print_pitch_deck(pitch_deck)
