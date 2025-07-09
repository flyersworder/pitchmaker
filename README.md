# TOO GOOD TO GO Pitch Maker 🍽️

An AI-powered application that helps TOO GOOD TO GO sales representatives create compelling, data-driven pitch decks and identify the most promising food business partnerships to reduce food waste.

## 🌟 Features

### 🔍 **Smart Food Business Search**
- **Google Places API Integration**: Search for any food business worldwide
- **AI-Powered Ranking**: Automatically evaluates and ranks businesses by TGTG partnership potential (0-10 scale)
- **Medal System**: 🥇🥈🥉 highlights for top 3 most promising prospects
- **Interactive Tooltips**: Hover over relevance scores to see detailed AI reasoning
- **Priority Indicators**: Color-coded badges (🟢 High, 🟡 Medium, ⚪ Low) for quick assessment

### 📊 **Comprehensive Pitch Deck Generation**
- **Evidence-Based Analysis**: Combines Google Places data with web research
- **Behavioral Science Integration**: Ready-to-use persuasion techniques with conversation scripts
- **Lead Temperature Assessment**: Visual indicators (🔥🔥🔥 Hot, 🔥🔥 Warm, ❄️ Cold)
- **Decision Maker Profiling**: Management style analysis and pain point identification
- **Sustainability Scoring**: Environmental impact assessment with detailed reasoning
- **Digital Readiness Evaluation**: Online presence and technology adoption analysis

### 🎯 **Sales-Optimized Output**
- **Phone Call Scripts**: Ready-to-use conversation starters and closing statements
- **Objection Handling**: Pre-crafted responses to common concerns
- **Contact Timing**: Optimal contact windows based on business operations
- **Download Options**: Export pitch decks as JSON or formatted text

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Google Maps API Key (Places API enabled)
- Google AI Studio API Key (Gemini)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pitchmaker
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   GOOGLE_API_KEY=your_google_ai_studio_api_key_here
   ```

4. **Run the Streamlit app**
   ```bash
   uv run streamlit run streamlit_app.py
   ```

5. **Access the application**
   Open your browser to `http://localhost:8501`

## 🛠️ Usage

### Food Business Search
1. Navigate to "🔍 Food Business Search"
2. Enter a search query (e.g., "Italian restaurants in Munich")
3. Wait for AI evaluation and ranking
4. Review results with TGTG relevance scores and reasoning

### Pitch Deck Generation
1. Navigate to "📊 Pitch Deck Generator"
2. Enter a food business query
3. Watch real-time processing logs
4. Review comprehensive pitch deck with:
   - Contact information
   - Decision maker profile
   - Key statistics and assessments
   - Behavioral science-based pitch strategies
   - Lead temperature and contact recommendations

### Command Line Interface
```bash
# Generate pitch deck from command line
uv run main.py --query "Help me prepare a pitch deck for Pak Choi in Munich"

# Debug mode
uv run main.py --verbose --query "your_query_here"

# Custom log level
uv run main.py --log-level DEBUG --query "your_query_here"
```

## 🏗️ Architecture

### Core Components

#### `main.py`
- **Core Logic**: Pitch deck generation engine
- **API Integration**: Google Places and Gemini AI
- **Data Models**: Pydantic schemas for structured output
- **CLI Interface**: Command-line tool with logging

#### `streamlit_app.py`
- **Web Interface**: User-friendly Streamlit application
- **AI Ranking**: LLM-powered TGTG relevance evaluation
- **Visual Design**: TOO GOOD TO GO branded styling
- **Interactive Features**: Real-time logs, tooltips, downloads

### Data Flow
1. **User Input** → Search query for food business
2. **Places API** → Retrieve business information
3. **Google Search** → Enrich with web research
4. **AI Analysis** → Generate insights and rankings
5. **Structured Output** → Formatted pitch deck
6. **Visual Display** → Interactive web interface

### AI Integration
- **Google Gemini 2.5 Flash**: Fast, accurate content generation
- **Structured Output**: Pydantic schema validation
- **Behavioral Science**: Evidence-based persuasion techniques
- **Partnership Assessment**: Custom scoring algorithms

## 📁 Project Structure

```
pitchmaker/
├── main.py                 # Core pitch deck generation logic
├── streamlit_app.py         # Web application interface
├── pyproject.toml         # Python dependencies
├── .env                     # Environment variables (create this)
├── .env.example            # Environment template
└── README.md               # This file
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_MAPS_API_KEY` | Google Maps Places API key | Yes |
| `GOOGLE_API_KEY` | Google AI Studio API key | Yes |

### API Requirements
- **Google Maps Platform**: Places API (Text Search)
- **Google AI Studio**: Gemini API access
- **Rate Limits**: Designed for moderate usage (100+ searches/day)

## 🎨 Customization

### Styling
- Brand colors defined in `streamlit_app.py` CSS section
- Modify TOO GOOD TO GO colors: `#00D68F` (primary), `#005A2D` (dark)
- Responsive design with mobile-friendly layout

### Assessment Criteria
- Modify relevance scoring in `evaluate_food_business_relevance()`
- Update TGTG partnership criteria in assessment prompts
- Customize behavioral science techniques in pitch generation

### Output Formats
- JSON schema defined by Pydantic models
- Text formatting in `print_pitch_deck()` function
- Add custom export formats as needed

## 🔍 Troubleshooting

### Common Issues

**API Key Errors**
```
Error: GOOGLE_MAPS_API_KEY environment variable not set
```
→ Check your `.env` file and API key configuration

**No Results Found**
```
No food businesses found for your search query
```
→ Try broader search terms or check business name spelling

**AI Evaluation Errors**
```
Error evaluating relevance: [error message]
```
→ Check Google AI Studio API key and quota limits

### Debug Mode
```bash
uv run main.py --verbose --query "test query"
```
Provides detailed logging for troubleshooting API calls and data processing.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow existing code style and structure
- Add tests for new features
- Update documentation for API changes
- Ensure all environment variables are documented

## 🙏 Acknowledgments

- **TOO GOOD TO GO**: For inspiring sustainable food waste reduction
- **Google**: Places API and Gemini AI platform
- **Streamlit**: Excellent web application framework
- **Pydantic**: Robust data validation and settings management

## 📞 Support

For issues, questions, or feature requests:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error logs and environment details

---

**Built with ❤️ for reducing food waste and creating sustainable partnerships**
