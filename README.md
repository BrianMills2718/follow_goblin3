# Twitter Network Analysis Tool

This Streamlit application allows you to analyze and visualize Twitter (X) following networks. It provides insights into community structures, influential accounts, and network relationships.

## Features

- Interactive network visualization (2D and 3D)
- Community detection and analysis
- Account importance metrics (PageRank and In-Degree)
- Tweet content analysis with AI-generated summaries
- Customizable filters for network exploration
- Beautiful and intuitive user interface

## Prerequisites

- Python 3.8 or higher
- Anaconda or Miniconda (recommended for environment management)
- RapidAPI account (for Twitter API access)
- OpenAI API key (for tweet analysis and community detection)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BrianMills2718/follow_goblin2.git
   cd follow_goblin2
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n nodegraph python=3.8
   conda activate nodegraph
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## API Key Setup

1. Create a `.streamlit` directory in your project folder:
   ```bash
   mkdir .streamlit
   ```

2. Create a `secrets.toml` file in the `.streamlit` directory with your API keys:
   ```toml
   RAPIDAPI_KEY = "your_rapidapi_key_here"
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

   To get your API keys:
   - RapidAPI Key: Sign up at [RapidAPI](https://rapidapi.com/) and subscribe to the Twitter API45 service
   - OpenAI API Key: Sign up at [OpenAI](https://platform.openai.com/) and create an API key

## Running the Application

1. Ensure you're in the project directory and your conda environment is activated:
   ```bash
   conda activate nodegraph
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Using the Application

1. **Basic Usage**:
   - Enter a Twitter username (without @) in the input field
   - Click "Generate Network" to analyze their following network
   - Use the sidebar controls to customize the visualization

2. **Display Options**:
   - Toggle between 2D and 3D visualization
   - Adjust node sizes and label sizes
   - Choose between PageRank or In-Degree for importance metrics

3. **Filtering**:
   - Filter by follower count, following count, and tweet count
   - Show/hide original node, first-degree, and second-degree connections
   - Filter by detected communities

4. **Community Analysis**:
   - Generate community labels based on account bios
   - Generate enhanced labels using tweet content
   - View top accounts in each community

5. **Tweet Analysis**:
   - Analyze tweets for top accounts
   - View AI-generated summaries of tweet content
   - Explore account statistics and metrics

## Troubleshooting

1. **API Key Issues**:
   - Ensure your API keys are correctly formatted in `secrets.toml`
   - Check that you have an active subscription to the required API services
   - Verify the `.streamlit` directory is in the correct location

2. **Network Issues**:
   - If the visualization is slow, try reducing the maximum number of accounts
   - For large networks, use the 2D visualization option
   - Ensure you have a stable internet connection for API calls

3. **Common Errors**:
   - "No secrets found": Check your `secrets.toml` file location and format
   - "API rate limit exceeded": Wait a few minutes before trying again
   - "Module not found": Ensure all requirements are installed in your conda environment

## Rate Limits and Usage

- RapidAPI Twitter service has rate limits (check your subscription plan)
- OpenAI API is used for tweet analysis and community detection (costs may apply)
- The application includes built-in rate limiting to prevent API overuse

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit
- Uses NetworkX for network analysis
- Powered by OpenAI's GPT models for text analysis
- Twitter data provided by RapidAPI's Twitter API45 service 