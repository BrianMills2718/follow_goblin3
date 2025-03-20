# ASYNCHRONOUS 3D NETWORK VISUALIZATION OF X ACCOUNT FOLLOWING NETWORK
#
# This script retrieves the "following" network of an input X (formerly Twitter) account asynchronously
# and visualizes it in an interactive 3D graph with permanent labels and detailed information tables.
#
# FEATURES:
#  1. 3D Force-Directed Graph visualization with:
#     - Permanent node labels that scale with node size
#     - Detailed hover information
#     - Interactive camera controls and node focusing
#  2. Node importance calculated using either CloutRank or In-Degree
#  3. Configurable node and label sizes
#  4. Display filters for:
#     - statuses_count, followers_count, friends_count, media_count
#     - created_at date range
#     - location (with search)
#     - verification status
#     - website presence
#     - business account status
#  5. Paginated tables showing:
#     - Top accounts by importance (CloutRank/In-Degree)
#     - Top independent accounts (not followed by original account)




import streamlit as st
import streamlit.components.v1 as components
import asyncio
import aiohttp
import json
from pyvis.network import Network
import datetime
from datetime import datetime as dt  # Import datetime as dt to avoid confusion
import numpy as np
from scipy import sparse
import google.generativeai as genai  # Updated import for Google Generative AI
from typing import List, Dict
import colorsys
import random
from tqdm import tqdm  # For progress tracking
import os
import pandas as pd
import networkx as nx

# Set page to wide mode - this must be the first Streamlit command
st.set_page_config(layout="wide", page_title="X Network Analysis", page_icon="🔍")

# CONSTANTS
RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]
RAPIDAPI_HOST = "twitter283.p.rapidapi.com"  # Updated API host

# Update OpenAI key to Gemini API key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # Use default if not in secrets
#OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
COMMUNITY_COLORS = {}  # Will be populated dynamically

async def get_following_async(screenname: str, session: aiohttp.ClientSession, cursor=None):
    """
    Asynchronously retrieve accounts that the given user is following.
    Updated to support pagination and the new API endpoint format.
    """
    endpoint = f"/FollowingLight?username={screenname}&count=20"
    if cursor and cursor != "-1":
        endpoint += f"&cursor={cursor}"
        
    url = f"https://{RAPIDAPI_HOST}{endpoint}"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
    
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                st.error(f"Failed to get following for {screenname}: {response.status}")
                return [], None
            data = await response.text()
            return parse_following_response(data)
    except Exception as e:
        st.error(f"Error fetching following: {str(e)}")
        return [], None

def parse_following_response(json_str):
    """Parse the JSON response from Twitter API for following accounts"""
    try:
        data = json.loads(json_str)
        
        accounts = []
        next_cursor = data.get("next_cursor_str")
        
        # Extract users from the FollowingLight response format
        users = data.get("users", [])
        
        for user in users:
            account = {
                "user_id": user.get("id_str"),
                "screen_name": user.get("screen_name", ""),
                "name": user.get("name", ""),
                "followers_count": user.get("followers_count", 0),
                "friends_count": user.get("friends_count", 0),
                "statuses_count": user.get("statuses_count", 0),
                "media_count": user.get("media_count", 0),
                "created_at": user.get("created_at", ""),
                "location": user.get("location", ""),
                "blue_verified": user.get("verified", False),  # API field changed
                "verified": user.get("verified", False),
                "website": user.get("url", ""),  # API field changed
                "business_account": False,  # Not available in this API
                "description": user.get("description", ""),
            }
            accounts.append(account)
        
        return accounts, next_cursor
    except Exception as e:
        st.error(f"Error parsing response: {str(e)}")
        return [], None

def compute_ratio(followers_count, friends_count):
    """Compute follower/following ratio; return 0 if denominator is zero."""
    return followers_count / friends_count if friends_count else 0




def compute_cloutrank(nodes, edges, damping=0.85, epsilon=1e-8, max_iter=100, return_contributors=False):
    """
    Compute CloutRank (PageRank) using the network structure with proper contribution tracking.
    
    Parameters:
    - nodes: Dict of node IDs to node data
    - edges: List of (source, target) edge tuples where source follows target
    - damping: Damping factor (default: 0.85)
    - epsilon: Convergence threshold (default: 1e-8)
    - max_iter: Maximum iterations (default: 100)
    - return_contributors: Whether to track and return contribution data
    
    Returns:
    - cloutrank_scores: Dict mapping node IDs to CloutRank scores
    - (optional) incoming_contributions: Dict mapping nodes to their contributors
    - (optional) outgoing_contributions: Dict mapping nodes to accounts they contribute to
    """
    # Construct the directed graph - edges point from follower to followed
    G = nx.DiGraph()
    G.add_nodes_from(nodes.keys())
    
    # Add edges, ensuring all node IDs are strings
    formatted_edges = [(str(src), str(tgt)) for src, tgt in edges]
    G.add_edges_from(formatted_edges)
    
    # Calculate in-degrees and out-degrees
    in_degrees = {node: G.in_degree(node) for node in G.nodes()}
    out_degrees = {node: G.out_degree(node) for node in G.nodes()}
    
    # Identify dangling nodes (nodes with no outgoing edges)
    dangling_nodes = [node for node, out_degree in out_degrees.items() if out_degree == 0]
    
    # Compute PageRank using NetworkX's implementation
    cloutrank_scores = nx.pagerank(G, alpha=damping, max_iter=max_iter, tol=epsilon)
    
    # If we don't need to track contributors, just return scores
    if not return_contributors:
        return cloutrank_scores
    
    # Initialize contribution tracking
    incoming_contributions = {node: {} for node in G.nodes()}  # Who contributes to me
    outgoing_contributions = {node: {} for node in G.nodes()}  # Who I contribute to
    
    # First, handle normal nodes (non-dangling)
    for node in G.nodes():
        # Skip dangling nodes, we'll handle them separately
        if node in dangling_nodes:
            continue
            
        # Get the node's PageRank score
        node_score = cloutrank_scores.get(node, 0)
        
        # Get all accounts this node follows
        followed_accounts = list(G.successors(node))
        
        # Calculate contribution per followed account
        contribution_per_followed = (damping * node_score) / len(followed_accounts)
        
        # Record contributions
        for followed in followed_accounts:
            incoming_contributions[followed][node] = contribution_per_followed
            outgoing_contributions[node][followed] = contribution_per_followed
    
    # Then handle dangling nodes - their PageRank is distributed to all nodes
    for node in dangling_nodes:
        node_score = cloutrank_scores.get(node, 0)
        
        # Dangling nodes distribute to all nodes equally
        contribution_per_node = (damping * node_score) / G.number_of_nodes()
        
        # Record these contributions
        for target in G.nodes():
            # Skip self-contributions for clarity
            if target == node:
                continue
                
            # Add dangling contribution
            if 'dangling' not in incoming_contributions[target]:
                incoming_contributions[target]['dangling'] = 0
            incoming_contributions[target]['dangling'] += contribution_per_node
            
            # Track outgoing from dangling node
            if 'global' not in outgoing_contributions[node]:
                outgoing_contributions[node]['global'] = 0
            outgoing_contributions[node]['global'] += contribution_per_node
    
    # Finally add random teleportation component
    teleport_weight = (1 - damping) / G.number_of_nodes()
    for node in G.nodes():
        # Each node gets teleportation weight from every node (including itself)
        for source in G.nodes():
            if 'teleport' not in incoming_contributions[node]:
                incoming_contributions[node]['teleport'] = 0
            incoming_contributions[node]['teleport'] += teleport_weight * cloutrank_scores.get(source, 0)
    
    return cloutrank_scores, incoming_contributions, outgoing_contributions



async def main_async(input_username: str, following_pages=2, second_degree_pages=1, fetch_tweets=False, tweet_pages=1):
    """
    Retrieves and processes the following network asynchronously with improved parallelization.
    Tweets are now fetched separately with the Summarize Tweets button.
    """
    nodes, edges = {}, []
    original_id = f"orig_{input_username}"
    
    # The original node: minimal attributes
    nodes[original_id] = {
        "screen_name": input_username,
        "name": input_username,
        "followers_count": None,
        "friends_count": None,
        "statuses_count": None,
        "media_count": None,
        "created_at": None,
        "location": None,
        "blue_verified": None,
        "verified": None,
        "website": None,
        "business_account": None,
        "ratio": None,
        "description": "",
        "direct": True,
        "tweets": [],
        "tweet_summary": ""
    }

    # Create progress indicators
    progress = st.progress(0)
    status_text = st.empty()
    
    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=50)) as session:
            # Step 1: Get following for original account with pagination
            status_text.text("Fetching accounts followed by original user...")
            first_hop_accounts = []
            cursor = None
            
            for page in range(following_pages):
                accounts, cursor = await get_following_async(input_username, session, cursor)
                first_hop_accounts.extend(accounts)
                
                progress.progress((page + 1) / (following_pages + 1))
                status_text.text(f"Fetched page {page+1}/{following_pages} of following accounts for original user")
                
                if not cursor or cursor == "-1":
                    break
            
            # Add first hop accounts to nodes and create edges
            for account in first_hop_accounts:
                uid = str(account.get("user_id"))
                if not uid:
                    continue
                
                ratio = compute_ratio(account.get("followers_count", 0), account.get("friends_count", 0))
                nodes[uid] = {
                    "screen_name": account.get("screen_name", ""),
                    "name": account.get("name", ""),
                    "followers_count": account.get("followers_count", 0),
                    "friends_count": account.get("friends_count", 0),
                    "statuses_count": account.get("statuses_count", 0),
                    "media_count": account.get("media_count", 0),
                    "created_at": account.get("created_at", ""),
                    "location": account.get("location", ""),
                    "blue_verified": account.get("blue_verified", False),
                    "verified": account.get("verified", False),
                    "website": account.get("website", ""),
                    "business_account": account.get("business_account", False),
                    "description": account.get("description", ""),
                    "ratio": ratio,
                    "direct": True,
                    "tweets": [],
                    "tweet_summary": ""
                }
                edges.append((original_id, uid))
            
            # Step 2: Get following for each first-degree account IN PARALLEL
            status_text.text(f"Fetching following for {len(first_hop_accounts)} first-degree accounts in parallel...")
            
            # Create tasks for all first-degree accounts
            second_degree_tasks = []
            for account in first_hop_accounts:
                source_id = str(account.get("user_id"))
                source_name = account.get("screen_name", "")
                
                # Create task to fetch second-degree connections
                task = fetch_second_degree_connections(source_id, source_name, session, second_degree_pages)
                second_degree_tasks.append(task)
            
            # Run all tasks concurrently with progress reporting
            total_tasks = len(second_degree_tasks)
            second_degree_results = []
            for i, task_coroutine in enumerate(asyncio.as_completed(second_degree_tasks), 1):
                result = await task_coroutine
                second_degree_results.append(result)
                progress.progress((following_pages + i) / (following_pages + total_tasks + 1))
                status_text.text(f"Processed {i}/{total_tasks} first-degree accounts")
            
            # Process all second-degree connections
            for source_id, connections in second_degree_results:
                for sid, node_data in connections:
                    if sid not in nodes:
                        nodes[sid] = node_data
                    edges.append((source_id, sid))
            
            # Complete progress
            progress.progress(1.0)
            status_text.text("Network data collection complete!")
            
    except Exception as e:
        st.error(f"Error in network collection: {str(e)}")
    
    return nodes, edges

# Helper function to fetch second-degree connections
async def fetch_second_degree_connections(source_id, source_name, session, max_pages):
    """Fetch all second-degree connections for a given account"""
    connections = []
    second_cursor = None
    
    for page in range(max_pages):
        accounts, second_cursor = await get_following_async(source_name, session, second_cursor)
        
        # Process accounts
        for account in accounts:
            sid = str(account.get("user_id"))
            if not sid:
                continue
            
            ratio = compute_ratio(account.get("followers_count", 0), account.get("friends_count", 0))
            node_data = {
                "screen_name": account.get("screen_name", ""),
                "name": account.get("name", ""),
                "followers_count": account.get("followers_count", 0),
                "friends_count": account.get("friends_count", 0),
                "statuses_count": account.get("statuses_count", 0),
                "media_count": account.get("media_count", 0),
                "created_at": account.get("created_at", ""),
                "location": account.get("location", ""),
                "blue_verified": account.get("blue_verified", False),
                "verified": account.get("verified", False),
                "website": account.get("website", ""),
                "business_account": account.get("business_account", False),
                "description": account.get("description", ""),
                "ratio": ratio,
                "direct": False,
                "tweets": [],
                "tweet_summary": ""
            }
            connections.append((sid, node_data))
        
        if not second_cursor or second_cursor == "-1":
            break
    
    return (source_id, connections)

# Helper function to fetch tweets and generate summaries
async def fetch_and_summarize_tweets(node_id, node, session, tweet_pages=1):
    """Fetch tweets and generate summary for an account - with optimizations"""
    username = node["screen_name"]
    
    try:
        # Fetch only one page of tweets for speed (20 tweets is enough for a summary)
        tweets, _ = await get_user_tweets_async(node_id, session, cursor=None)
        
        # Generate summary if tweets were found
        summary = "No tweets available"
        if tweets:
            summary = await generate_tweet_summary(tweets, username)
        
        return (node_id, tweets, summary)
    except Exception as e:
        st.error(f"Error processing tweets for @{username}: {str(e)}")
        return (node_id, [], f"Error fetching tweets: {str(e)}")

def filter_nodes(nodes, filters):
    """
    Filters nodes based on provided filter criteria.
    """
    filtered = {}
    for node_id, node in nodes.items():
        # Always include the original node.
        if node_id.startswith("orig_"):
            filtered[node_id] = node
            continue

        # Helper function to safely compare values that might be None
        def is_in_range(value, min_val, max_val):
            if value is None:
                return False
            return min_val <= value <= max_val

        # Numeric filters with None handling
        if not is_in_range(node.get("statuses_count"), filters["statuses_range"][0], filters["statuses_range"][1]):
            continue
        if not is_in_range(node.get("followers_count"), filters["followers_range"][0], filters["followers_range"][1]):
            continue
        if not is_in_range(node.get("friends_count"), filters["friends_range"][0], filters["friends_range"][1]):
            continue
        if not is_in_range(node.get("media_count"), filters["media_range"][0], filters["media_range"][1]):
            continue

        # Location filters
        location = node.get("location")
        if filters["selected_locations"]:
            if location is not None and isinstance(location, str) and location.strip():
                location = location.strip().lower()
                if not any(loc.lower() in location for loc in filters["selected_locations"]):
                    continue
            else:
                continue
        elif filters["require_location"]:
            if not location or not isinstance(location, str) or not location.strip():
                continue

        # Blue verified filter.
        if filters["require_blue_verified"]:
            if not node.get("blue_verified", False):
                continue

        # Verified filter.
        if filters["verified_option"] == "Only Verified":
            if not node.get("verified", False):
                continue
        elif filters["verified_option"] == "Only Not Verified":
            if node.get("verified", False):
                continue

        # Website filter.
        if filters["require_website"]:
            if not node.get("website", "").strip():
                continue

        # Business account filter.
        if filters["business_account_option"] == "Only Business Accounts":
            if not node.get("business_account", False):
                continue
        elif filters["business_account_option"] == "Only Non-Business Accounts":
            if node.get("business_account", False):
                continue
        
        filtered[node_id] = node
    return filtered

# ---------------------------------------------------------------------
# NEW: Build a 3D network visualization using ForceGraph3D.
# ---------------------------------------------------------------------
def get_openai_client():
    """Initialize Gemini client."""
    # Configure the Gemini API key
    genai.configure(api_key=GEMINI_API_KEY)
    # Return the Gemini model
    return genai.GenerativeModel("gemini-2.0-flash")

async def get_community_labels(accounts: List[Dict]) -> List[str]:
    """Get community labels from Gemini using account descriptions."""
    client = get_openai_client()
    
    # Create prompt with account information
    account_info = "\n".join([
        f"Username: {acc['screen_name']}, Description: {acc['description']}"
        for acc in accounts
    ])
    
    prompt = f"""Analyze these X/Twitter accounts and their descriptions. Create community labels that provide good coverage of the different types of accounts present. Include an "Other" category for accounts that don't fit well into specific groups.

Accounts to analyze:
{account_info}

Return your response as a JSON object mapping community IDs to labels, like:
{{
  "0": "Tech Entrepreneurs",
  "1": "Political Commentators", 
  "2": "Other"
}}"""

    try:
        response = client.generate_content(prompt)
        
        # Parse response into dictionary of labels
        response_text = response.text.strip()
        import re
        json_match = re.search(r'({[\s\S]*})', response_text)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        return {}
    except Exception as e:
        st.error(f"Error generating community labels: {str(e)}")
        return {}

async def classify_accounts(accounts: List[Dict], labels: List[str], batch_size=50) -> Dict[str, str]:
    """Classify accounts into communities in parallel batches."""
    client = get_openai_client()
    results = {}
    
    # Create a container for progress indicators
    progress_container = st.container()
    with progress_container:
        st.write("### Community Classification Progress")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        batch_status = st.empty()
        
        # Show initial stats
        total_batches = (len(accounts) + batch_size - 1) // batch_size
        st.write(f"Total accounts to process: {len(accounts)}")
        st.write(f"Number of batches: {total_batches}")
        
        # Create a placeholder for batch status updates
        batch_updates = st.empty()
        completed_batches = set()

    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(10)  # Process up to 10 batches concurrently
    
    async def process_batch(batch, batch_num):
        async with semaphore:  # Limit concurrent processing
            accounts_info = "\n".join([
                f"Username: {acc['screen_name']}, Description: {acc['description']}"
                for acc in batch
            ])
            
            labels_str = "\n".join(labels)
            prompt = f"""Given these community labels:
{labels_str}

Classify each of these accounts into exactly one of the above communities.
Only use the exact community labels provided above, do not create new ones.
If unsure, use the 'Other' category.

Accounts to classify:
{accounts_info}

Return in format:
username: community_label"""

            try:
                # Update status before processing
                with batch_status:
                    st.write(f"Processing batch {batch_num + 1}...")
                
                # Initialize Gemini client
                genai.configure(api_key=GEMINI_API_KEY)
                client = genai.GenerativeModel("gemini-2.0-flash")
                
                # Generate classifications
                response = client.generate_content(prompt)
                
                # Parse response into dictionary
                classifications = {}
                for line in response.text.split('\n'):
                    if ':' in line:
                        username, label = line.split(':', 1)
                        label = label.strip()
                        # Only store valid community labels
                        if label in labels:
                            classifications[username.strip()] = label
                        else:
                            # If invalid label, use "Other"
                            classifications[username.strip()] = "Other"
                
                completed_batches.add(batch_num)
                
                # Update batch status
                with batch_updates:
                    st.write(f"✅ Completed batch {batch_num + 1}/{total_batches}")
                
                return batch_num, classifications
            except Exception as e:
                with batch_updates:
                    st.error(f"❌ Error in batch {batch_num + 1}: {str(e)}")
                return batch_num, {}

    # Create batches with larger batch size
    batches = []
    for i in range(0, len(accounts), batch_size):
        batch = accounts[i:i + batch_size]
        batches.append((batch, i // batch_size))
    
    # Process all batches in parallel with high concurrency
    tasks = [process_batch(batch, batch_num) for batch, batch_num in batches]
    batch_results = await asyncio.gather(*tasks)
    
    # Update progress as results come in
    for batch_num, classifications in sorted(batch_results, key=lambda x: x[0]):
        results.update(classifications)
        progress = (batch_num + 1) / total_batches
        progress_text.text(f"Overall Progress: {progress:.1%}")
        progress_bar.progress(progress)
    
    # Show final status
    with progress_container:
        if len(completed_batches) == total_batches:
            st.success(f"✅ Classification complete! Processed {len(accounts)} accounts in {total_batches} batches.")
        else:
            st.warning(f"⚠️ Classification partially complete. {len(completed_batches)}/{total_batches} batches processed.")
    
    return results

def generate_n_colors(n):
    """Generate n visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + random.uniform(-0.2, 0.2)
        value = 0.9 + random.uniform(-0.2, 0.2)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255))
        colors.append(hex_color)
    return colors

def build_network_3d(nodes, edges, max_nodes=10, size_factors=None, use_pagerank=False):
    """
    Constructs a 3D ForceGraph visualization with permanent labels and hover info.
    Updated to apply max_nodes selection similar to the build_network_2d function.
    """
    # Debug info
    st.write(f"DEBUG: Building 3D network with {len(nodes)} nodes, max_nodes={max_nodes}")
    
    # Set default size factors if None
    if size_factors is None:
        size_factors = {
            'base_size': 5,
            'importance_factor': 3.0,
            'label_size_factor': 1.0
        }

    # Determine node importance - ALWAYS calculate both metrics
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
            
    cloutrank = compute_cloutrank(nodes, edges)
    
    # Choose which metric to use for sizing nodes
    importance = cloutrank if use_pagerank else in_degrees

    # Identify the original node
    original_id = next(id for id in nodes.keys() if id.startswith("orig_"))
    followed_by_original = {tgt for src, tgt in edges if src == original_id}

    # Select top nodes based on the importance score
    top_overall = sorted(
        [(nid, score) for nid, score in importance.items() if not nid.startswith("orig_")],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes // 2]

    top_independent = sorted(
        [(nid, score) for nid, score in importance.items()
         if not nid.startswith("orig_") and nid not in followed_by_original],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes // 2]

    # Create the selected node set
    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}
    
    # Debug info - important for troubleshooting
    st.write(f"DEBUG: Selected {len(selected_nodes)} nodes for visualization (requested max: {max_nodes})")
    
    # Store the number of nodes selected in session state for debugging
    st.session_state.last_selected_node_count = len(selected_nodes)
    st.session_state.last_max_nodes = max_nodes

    # Filter nodes and edges to only include selected nodes
    filtered_nodes = {node_id: meta for node_id, meta in nodes.items() if node_id in selected_nodes}
    filtered_edges = [(src, tgt) for src, tgt in edges if src in selected_nodes and tgt in selected_nodes]

    nodes_data = []
    links_data = []

    # Convert edges to proper format
    links_data = [{"source": str(src), "target": str(tgt)} for src, tgt in filtered_edges]

    # Convert nodes to proper format with additional info
    for node_id, meta in filtered_nodes.items():
        try:
            base_size = float(size_factors.get('base_size', 5))
            importance_factor = float(size_factors.get('importance_factor', 3.0))
            
            # Handle None values for followers_count
            followers_count = meta.get("followers_count")
            if followers_count is None:
                followers_count = 0 if node_id.startswith("orig_") else 1000  # Default value for non-original nodes
            
            # Calculate node size with type checking
            followers_factor = float(followers_count) / 1000.0
            node_size = base_size + followers_factor * importance_factor
            
            # Ensure node_size is positive
            node_size = max(1.0, node_size)
            
            # Handle None values for other metrics
            following_count = meta.get("friends_count", 0)
            if following_count is None:
                following_count = 0
                
            ratio = meta.get("ratio", 0.0)
            if ratio is None:
                ratio = 0.0
                
            # Get community color if it exists
            community_color = meta.get("community_color", "#6ca6cd")  # Use default color if no community
            
            # Get community information
            username = meta.get("screen_name", "")
            community = "N/A"
            if ('node_communities' in st.session_state and 
                st.session_state.node_communities and 
                username in st.session_state.node_communities):
                community = st.session_state.node_communities[username]
                # CHANGE: Translate community ID to descriptive label
                if ('community_labels' in st.session_state and 
                    st.session_state.community_labels and 
                    community in st.session_state.community_labels):
                    community = st.session_state.community_labels[community]
            
            # FIX: Always store actual CloutRank and In-Degree values separately
            nodes_data.append({
                "id": str(node_id),
                "name": str(meta.get("screen_name", "")),
                "community": community,
                "followers": int(followers_count),
                "following": int(following_count),
                "ratio": float(ratio),
                "size": float(node_size),
                "description": str(meta.get("description", "")),
                "cloutrank": float(cloutrank.get(node_id, 0)),  # Always use true CloutRank
                "indegree": int(in_degrees.get(node_id, 0)),  # Always use true In-Degree
                "importance": float(importance.get(node_id, 0)),  # Selected importance metric
                "color": community_color
            })
        except Exception as e:
            st.write(f"Warning: Error processing node {node_id}: {str(e)}")
            nodes_data.append({
                "id": str(node_id),
                "name": str(meta.get("screen_name", "")),
                "community": "N/A",
                "followers": 0,
                "following": 0,
                "ratio": 0.0,
                "size": float(size_factors.get('base_size', 5)),
                "description": "",
                "cloutrank": 0.0,
                "indegree": 0,
                "importance": 0.0,
                "color": "#6ca6cd"
            })

    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links_data)

    # Update tooltip to correctly show both metrics
    importance_label = "CloutRank" if use_pagerank else "In-Degree"

    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <script src="https://unpkg.com/three@0.149.0/build/three.min.js"></script>
        <script src="https://unpkg.com/3d-force-graph@1.70.10/dist/3d-force-graph.min.js"></script>
        <script src="https://unpkg.com/three-spritetext@1.6.5/dist/three-spritetext.min.js"></script>
        <style>
          #graph {{ width: 100%; height: 750px; }}
          .node-tooltip {{
              font-family: Arial;
              padding: 8px;
              border-radius: 4px;
              background-color: rgba(0,0,0,0.8);
              color: white;
              white-space: pre-line;
              font-size: 14px;
          }}
        </style>
      </head>
      <body>
        <div id="graph"></div>
        <script>
          const data = {{
            nodes: {nodes_json},
            links: {links_json}
          }};
          
          console.log("Graph data:", data);
          
          const Graph = ForceGraph3D()
            (document.getElementById('graph'))
            .graphData(data)
            .nodeColor(node => node.color)
            .nodeRelSize(6)
            .nodeVal(node => node.size)
            .nodeThreeObject(node => {{
                const group = new THREE.Group();
                
                const sphere = new THREE.Mesh(
                    new THREE.SphereGeometry(Math.cbrt(node.size)),
                    new THREE.MeshLambertMaterial({{
                        color: node.color,
                        transparent: true,
                        opacity: 0.75
                    }})
                );
                group.add(sphere);
                
                const sprite = new SpriteText(node.name);
                sprite.textHeight = Math.max(4, Math.min(12, 8 * Math.cbrt(node.size / 10))) * 
                                  {size_factors.get('label_size_factor', 1.0)};
                sprite.color = 'white';
                sprite.backgroundColor = 'rgba(0,0,0,0.6)';
                sprite.padding = 2;
                sprite.borderRadius = 3;
                sprite.position.y = Math.cbrt(node.size) + 1;
                group.add(sprite);
                
                return group;
            }})
            .nodeLabel(node => {{
                return `<div class="node-tooltip">
                    <b>@${{node.name}}</b><br/>
                    Community: ${{node.community}}<br/>
                    Followers: ${{node.followers.toLocaleString()}}<br/>
                    Following: ${{node.following.toLocaleString()}}<br/>
                    Ratio: ${{node.ratio.toFixed(2)}}<br/>
                    Description: ${{node.description}}<br/>
                    ${importance_label}: ${{node.importance.toFixed(4)}}<br/>
                    CloutRank: ${{node.cloutrank.toFixed(4)}}
                    </div>`;
            }})
            .linkDirectionalParticles(1)
            .linkDirectionalParticleSpeed(0.006)
            .backgroundColor("#101020");

          // Set initial camera position
          Graph.cameraPosition({{ x: 150, y: 150, z: 150 }});

          // Adjust force parameters for better layout
          Graph.d3Force('charge').strength(-120);
          
          // Add node click behavior for camera focus
          Graph.onNodeClick(node => {{
              const distance = 40;
              const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);
              Graph.cameraPosition(
                  {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
                  node,
                  2000
              );
          }});
        </script>
      </body>
    </html>
    """
    return html_code


# ---------------------------------------------------------------------
# End new 3D visualization code
# ---------------------------------------------------------------------

def build_network_2d(nodes, edges, max_nodes=10, size_factors=None, use_pagerank=False):
    """
    Constructs a 2D network visualization using pyvis.
    Uses the same parameters as build_network_3d for consistency.
    """
    # Change default from use_pagerank=False to use_clout=True or similar parameter
    
    # Compute importance scores based on PageRank or Clout
    if use_pagerank:
        importance_scores = compute_pagerank(nodes, edges)
    else:
        importance_scores = compute_cloutrank(nodes, edges)
    
    # Find original node and followed nodes
    original_id = next(id for id in nodes.keys() if id.startswith("orig_"))
    followed_by_original = {tgt for src, tgt in edges if src == original_id}
    
    # Select nodes same way as 3D version
    top_overall = sorted(
        [(nid, score) for nid, score in importance_scores.items() 
         if not nid.startswith("orig_")],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]
    
    top_independent = sorted(
        [(nid, score) for nid, score in importance_scores.items() 
         if not nid.startswith("orig_") and nid not in followed_by_original],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]
    
    # Create the selected node set
    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}
    
    # Store the number of nodes selected in session state for debugging
    st.session_state.last_selected_node_count_2d = len(selected_nodes)
    st.session_state.last_max_nodes_2d = max_nodes

    # Create pyvis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Normalize importance scores
    max_importance = max(importance_scores.values())
    normalized_importance = {nid: score/max_importance for nid, score in importance_scores.items()}
    
    # Add nodes
    for node_id in selected_nodes:
        size = (size_factors['base_size'] +
                normalized_importance[node_id] * size_factors['importance_factor'] * 20)
        
        # Safely format numeric values by checking for None
        followers = nodes[node_id].get('followers_count')
        followers_str = f"{followers:,}" if isinstance(followers, int) else "0"
        friends = nodes[node_id].get('friends_count')
        friends_str = f"{friends:,}" if isinstance(friends, int) else "0"
        ratio = nodes[node_id].get('ratio')
        ratio_str = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "0.00"
        description = nodes[node_id].get('description') or ""
        
        # Include both metrics in the tooltip
        importance_label = "CloutRank" if use_pagerank else "In-Degree"
        importance_value = importance_scores.get(node_id, 0)
        importance_str = f"{importance_value:.4f}" if isinstance(importance_value, float) else f"{importance_value}"
        
        cloutrank_value = compute_cloutrank(nodes, edges).get(node_id, 0)
        cloutrank_str = f"{cloutrank_value:.4f}" if isinstance(cloutrank_value, float) else "0.0000"
        
        indegree_value = in_degrees.get(node_id, 0)
        
        title = (f"Followers: {followers_str}\n"
                 f"Following: {friends_str}\n"
                 f"Ratio: {ratio_str}\n"
                 f"Current Importance ({importance_label}): {importance_str}\n"
                 f"CloutRank: {cloutrank_str}\n"
                 f"In-Degree: {indegree_value}\n"
                 f"Description: {description}")

        color = nodes[node_id].get("community_color", "#6ca6cd")

        net.add_node(
            node_id,
            label=nodes[node_id]["screen_name"],
            title=title,
            size=size,
            color=color
        )
    
    # Add edges between selected nodes
    for src, tgt in edges:
        if src in selected_nodes and tgt in selected_nodes:
            net.add_edge(src, tgt)
    
    return net

def create_account_table(accounts_data, start_idx=0, page_size=10, include_tweets=False):
    """
    Create a table for displaying account information.
    accounts_data should be a list of tuples: (node_id, score, node_data)
    """
    if not accounts_data:
        return st.write("No accounts to display")
    
    # Update table columns - remove F/F Ratio
    table_data = {
        "Rank": [],
        "Username": [],
        "Score": [],
        "Followers": [],
        "Following": [],
        "Description": []
    }
    
    # Add tweet summary column if requested
    if include_tweets:
        table_data["Tweet Summary"] = []
        table_data["Fetch Status"] = []  # Add column for fetch status
    
    end_idx = min(start_idx + page_size, len(accounts_data))
    for idx, (_, score, node) in enumerate(accounts_data[start_idx:end_idx], start=start_idx + 1):
        table_data["Rank"].append(idx)
        table_data["Username"].append(node["screen_name"])
        table_data["Score"].append(f"{score:.4f}")
        table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
        table_data["Following"].append(f"{node.get('friends_count', 0):,}")
        table_data["Description"].append(node.get("description", ""))
        
        # Add tweet summary if requested
        if include_tweets:
            table_data["Tweet Summary"].append(node.get("tweet_summary", "No summary available"))
            # Add fetch status if available
            fetch_status = node.get("tweet_fetch_status", "")
            table_data["Fetch Status"].append(fetch_status)
    
    st.table(table_data)
    
    return end_idx < len(accounts_data)

def run_async_main(input_username: str, following_pages=2, second_degree_pages=1, fetch_tweets=False, tweet_pages=1):
    """Wrapper to execute the asynchronous function with pagination and tweet options."""
    return asyncio.run(main_async(input_username, following_pages, second_degree_pages, fetch_tweets, tweet_pages))

def get_top_accounts_by_community(nodes: Dict, node_communities: Dict, importance_scores: Dict, top_n: int = 10) -> Dict[str, List]:
    """Get top accounts for each community based on importance scores."""
    community_accounts = {}
    
    # Group accounts by community
    for node_id, node in nodes.items():
        if node_id.startswith("orig_"):
            continue
            
        username = node["screen_name"]
        if username in node_communities:
            community = node_communities[username]
            if community not in community_accounts:
                community_accounts[community] = []
            community_accounts[community].append((node_id, node, importance_scores.get(node_id, 0)))
    
    # Sort accounts within each community by importance
    top_accounts = {}
    for community, accounts in community_accounts.items():
        sorted_accounts = sorted(accounts, key=lambda x: x[2], reverse=True)[:top_n]
        top_accounts[community] = sorted_accounts
    
    return top_accounts

def display_community_tables(top_accounts_by_community, colors, filtered_nodes, edges, include_tweets=False):
    """Display tables for top accounts in each community."""
    
    # Calculate in-degrees
    in_degrees = {node_id: 0 for node_id in filtered_nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
    
    # Use pre-computed cloutrank if available
    if 'importance_scores' in st.session_state and isinstance(next(iter(st.session_state.importance_scores.values()), 0), float):
        cloutrank = st.session_state.importance_scores
    else:
        # Calculate on demand only if we have to
        cloutrank = compute_cloutrank(filtered_nodes, edges)
    
    for community_id, accounts in top_accounts_by_community.items():
        # Skip if no accounts
        if not accounts:
            continue
        
        # Get community color and label
        color = colors.get(community_id, "#cccccc")
        
        # Use descriptive label instead of just "Community X"
        if 'community_labels' in st.session_state and st.session_state.community_labels:
            label = st.session_state.community_labels.get(community_id, f"Community {community_id}")
        else:
            label = f"Community {community_id}"
        
        # Style the community header with its color
        st.markdown(
            f"<h3 style='color:{color}'>{label}</h3>", 
            unsafe_allow_html=True
        )
        
        # Create table data with SAME COLUMNS as top_accounts_table
        table_data = {
            "Rank": [],
            "Username": [],
            "CloutRank": [],        # Always include CloutRank
            "In-Degree": [],        # Always include In-Degree
            "Followers": [],
            "Following": [],
            "Description": []
        }
        
        # Add tweet summary column if requested
        if include_tweets:
            table_data["Tweet Summary"] = []
        
        # Only show top 10 per community
        show_accounts = accounts[:10]
        
        # Populate table
        for idx, (node_id, node, score) in enumerate(show_accounts, 1):
            table_data["Rank"].append(idx)
            table_data["Username"].append(node["screen_name"])
            
            # Always add both metrics separately
            cr_value = cloutrank.get(node_id, 0)
            table_data["CloutRank"].append(f"{cr_value:.4f}")
            
            id_value = in_degrees.get(node_id, 0)
            table_data["In-Degree"].append(str(id_value))
            
            # Add other account data
            table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
            table_data["Following"].append(f"{node.get('friends_count', 0):,}")
            table_data["Description"].append(node.get("description", ""))
        
            # Add tweet summary if requested
            if include_tweets:
                # Get tweet summary with improved formatting
                tweet_summary = node.get("tweet_summary", "")
                if tweet_summary and len(tweet_summary) > 0:
                    # Format it nicely
                    table_data["Tweet Summary"].append(tweet_summary)
                else:
                    table_data["Tweet Summary"].append("No tweet summary available")
        
        # Display table
        st.table(table_data)

def display_top_accounts_table(nodes, edges, importance_scores, original_id, exclude_first_degree=False, include_tweets=False):
    """Display table of top accounts based on importance scores."""
    # Calculate in-degrees
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
    
    # Use pre-computed full cloutrank if available
    if 'full_cloutrank' in st.session_state and st.session_state.importance_metric_mode == "CloutRank":
        # Use the pre-computed scores from the complete graph
        cloutrank = st.session_state.full_cloutrank
    else:
        # If we're not in CloutRank mode or don't have pre-computed scores, use current scores
        if isinstance(next(iter(importance_scores.values()), 0), float):
            cloutrank = importance_scores
        else:
            # Only calculate if absolutely necessary
            cloutrank = compute_cloutrank(nodes, edges)
    
    # Get metric name for header
    importance_metric = st.session_state.get("importance_metric_mode", "In-Degree")
    
    st.subheader(f"Top {min(20, len(nodes)-1)} Accounts by {importance_metric}")
    
    # Get direct follows for filtering
    direct_follows = set()
    for src, tgt in edges:
        if src == original_id:
            direct_follows.add(tgt)
    
    # Filter and sort accounts
    accounts = []
    for node_id, node in nodes.items():
        # Skip original account and optionally first-degree connections
        if node_id == original_id or (exclude_first_degree and node_id in direct_follows):
            continue
    
        score = importance_scores.get(node_id, 0)
        accounts.append((node_id, node, score))
    
    # Sort by importance score
    accounts.sort(key=lambda x: x[2], reverse=True)
    
    # Only show top 20
    accounts = accounts[:20]
    
    # Create table data with both metrics always included
    table_data = {
        "Rank": [],
        "Username": [],
        "CloutRank": [],        # Always include CloutRank
        "In-Degree": [],        # Always include In-Degree
        "Followers": [],
        "Following": [],
        "Description": []
    }
    
    # Only add community column if communities exist
    communities_exist = ('node_communities' in st.session_state and st.session_state.node_communities)
    if communities_exist:
        table_data["Community"] = []
    
    # Add tweet summary column if requested
    if include_tweets:
        table_data["Tweet Summary"] = []
    
    # Get communities if available
    communities = st.session_state.get("node_communities", {}) or {}
    community_labels = st.session_state.get("community_labels", {}) or {}
    
    # Populate table
    for idx, (node_id, node, score) in enumerate(accounts, 1):
        table_data["Rank"].append(idx)
        table_data["Username"].append(node["screen_name"])
        
        # Add community data if communities exist
        if communities_exist:
            community_id = communities.get(node["screen_name"], "0")
            community_label = community_labels.get(community_id, f"Community {community_id}")
            table_data["Community"].append(community_label)
        
        # Always add both metrics separately
        cr_value = cloutrank.get(node_id, 0)
        table_data["CloutRank"].append(f"{cr_value:.4f}")
        
        id_value = in_degrees.get(node_id, 0)
        table_data["In-Degree"].append(str(id_value))
        
        # Add other account data
        table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
        table_data["Following"].append(f"{node.get('friends_count', 0):,}")
        table_data["Description"].append(node.get("description", ""))
    
        # Add tweet summary if requested
        if include_tweets:
            tweet_summary = node.get("tweet_summary", "")
            if tweet_summary and len(tweet_summary) > 0:
                table_data["Tweet Summary"].append(tweet_summary)
            else:
                table_data["Tweet Summary"].append("No tweet summary available")
    
    # Display table
    st.table(table_data)

def create_downloadable_account_table(nodes, edges, include_tweets=False, include_communities=False):
    """Create a comprehensive downloadable table with all account information."""
    
    # Calculate in-degrees for all nodes
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
    
    # Debug information about the input
    st.write(f"Debug: create_downloadable_account_table called with {len(nodes)} nodes and {len(edges)} edges")
    
    # Use pre-computed cloutrank if available, or calculate with contributors
    if 'full_cloutrank' in st.session_state and st.session_state.importance_metric_mode == "CloutRank":
        st.write("Debug: Using pre-computed CloutRank scores")
        # If we have the cached scores but not contributors, recalculate with contributors
        if 'cloutrank_contributions' not in st.session_state:
            st.write("Debug: No cached contributions found, recalculating...")
            cloutrank, incoming_contribs, outgoing_contribs = compute_cloutrank(nodes, edges, return_contributors=True)
            st.session_state.cloutrank_contributions = (incoming_contribs, outgoing_contribs)
        else:
            st.write("Debug: Using cached CloutRank contributions")
            cloutrank = st.session_state.full_cloutrank
            incoming_contribs, outgoing_contribs = st.session_state.cloutrank_contributions
    else:
        st.write("Debug: Computing fresh CloutRank scores and contributions")
        # Calculate from scratch with contributors
        cloutrank, incoming_contribs, outgoing_contribs = compute_cloutrank(nodes, edges, return_contributors=True)
        st.session_state.cloutrank_contributions = (incoming_contribs, outgoing_contribs)
    
    st.write(f"Debug: Have contributions for {len(incoming_contribs)} nodes")
    st.write(f"Debug: Total incoming contributions: {sum(len(c) for c in incoming_contribs.values())}")
    st.write(f"Debug: Total outgoing contributions: {sum(len(c) for c in outgoing_contribs.values())}")
    
    # Get communities if available
    communities = st.session_state.get("node_communities", {}) or {}
    community_labels = st.session_state.get("community_labels", {}) or {}
    
    # Create DataFrame for all accounts
    data = []
    processed_nodes = 0
    nodes_with_contributions = 0
    
    for node_id, node in nodes.items():
        # Skip nodes that might not be complete accounts
        if not isinstance(node, dict) or "screen_name" not in node:
            continue
        
        processed_nodes += 1
        
        # Get community info if available
        community_id = communities.get(node["screen_name"], "")
        community_label = community_labels.get(community_id, "") if community_id else ""
        
        # Get tweet summary
        tweet_summary = node.get("tweet_summary", "") if include_tweets else ""
        
        # Calculate ratio
        followers = node.get("followers_count", 0)
        following = node.get("friends_count", 0)
        ratio = compute_ratio(followers, following)
        
        # Format incoming contributors list
        node_incoming = incoming_contribs.get(node_id, {})
        sorted_incoming = sorted(
            node_incoming.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        incoming_str = "; ".join([
            f"{nodes[src]['screen_name']} ({contribution:.2f})"
            for src, contribution in sorted_incoming[:5]  # Show top 5 contributors
            if src in nodes
        ])
        if len(sorted_incoming) > 5:
            incoming_str += f"; ... ({len(sorted_incoming)-5} more)"
        
        # Format outgoing contributions list
        node_outgoing = outgoing_contribs.get(node_id, {})
        sorted_outgoing = sorted(
            node_outgoing.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        outgoing_str = "; ".join([
            f"{nodes[tgt]['screen_name']} ({contribution:.2f})"
            for tgt, contribution in sorted_outgoing[:5]  # Show top 5 contributions
            if tgt in nodes
        ])
        if len(sorted_outgoing) > 5:
            outgoing_str += f"; ... ({len(sorted_outgoing)-5} more)"
        
        if incoming_str or outgoing_str:
            nodes_with_contributions += 1
        
        # Add row to data
        row = {
            "Screen Name": node["screen_name"],
            "Name": node.get("name", ""),
            "CloutRank": cloutrank.get(node_id, 0),
            "In-Degree": in_degrees.get(node_id, 0),
            "Followers": followers,
            "Following": following,
            "Ratio": ratio,
            "Tweets": node.get("statuses_count", 0),
            "Media": node.get("media_count", 0),
            "Created At": node.get("created_at", ""),
            "Verified": node.get("verified", False),
            "Blue Verified": node.get("blue_verified", False),
            "Business Account": node.get("business_account", False),
            "Website": node.get("website", ""),
            "Location": node.get("location", ""),
            "Community": community_label,
            "Incoming CloutRank Contributions": incoming_str,
            "Outgoing CloutRank Contributions": outgoing_str,
            "Tweet Summary": tweet_summary
        }
        data.append(row)
    
    st.write(f"Debug: Processed {processed_nodes} nodes, {nodes_with_contributions} had contributions")
    
    # Convert to DataFrame and sort by CloutRank
    df = pd.DataFrame(data)
    df = df.sort_values("CloutRank", ascending=False)
    
    return df

def main():
    # Reset community colors if it's not a dictionary
    if 'community_colors' in st.session_state and not isinstance(st.session_state.community_colors, dict):
        st.session_state.pop('community_colors')
    
    # Initialize session state variables if they don't exist
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    if 'community_labels' not in st.session_state:
        st.session_state.community_labels = None
    if 'community_colors' not in st.session_state:
        st.session_state.community_colors = None
    if 'node_communities' not in st.session_state:
        st.session_state.node_communities = None
    if 'use_3d' not in st.session_state:
        st.session_state.use_3d = True
    if 'cloutrank_contributors' not in st.session_state:
        st.session_state.cloutrank_contributors = None

    st.title("X Account Following Network Visualization")
    st.markdown("Enter an X (formerly Twitter) username to retrieve its following network.")

    input_username = st.text_input("X Username (without @):", value="elonmusk")
    
    # Sidebar: Display Options and Filter Criteria
    st.sidebar.header("Display Options")
    
    # Node and Label Size Controls
    st.sidebar.subheader("Size Controls")
    
    # Dropdown menu for importance metric
    importance_metric = st.sidebar.selectbox(
        "Importance Metric", 
        options=["In-Degree", "CloutRank"],
        index=0,
        help="In-Degree measures importance by how many accounts follow this account in the network. CloutRank considers both quantity and quality of connections."
    )
    use_pagerank = (importance_metric == "CloutRank")
    
    # Replace slider with number_input for direct typing
    account_size_factor = st.sidebar.number_input(
        "Account Size Factor",
        min_value=0.1,
        max_value=10.0,
        value=3.0,
        step=0.1,
        format="%.1f",
        help="Controls how much account importance affects node size in the visualization. Higher values make important accounts appear larger."
    )
    
    # Label size control with number_input
    label_size = 1.0  # Default value
    if st.session_state.network_data is not None:
        st.session_state.use_3d = st.checkbox(
            "Use 3D Visualization", 
            value=st.session_state.use_3d,
            help="Toggle between 3D and 2D network visualization. 3D offers more interactive features but may be slower with large networks."
        )
        if st.session_state.use_3d:
            label_size = st.sidebar.number_input(
                "Label Size",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                help="Controls the size of the account name labels in the 3D visualization."
            )
    
    # Number input for max accounts
    max_accounts_display = st.sidebar.number_input(
        "Max Accounts to Display",
        min_value=5,
        max_value=1000,
        value=st.session_state.get('max_accounts_display', 50),  # Use session state to remember
        step=5,
        help="Maximum number of accounts to show in the visualization. Lower values improve performance."
    )
    
    # Store in session state for persistence
    st.session_state['max_accounts_display'] = max_accounts_display

    # Add a button to update the visualization - use a SINGLE button with a unique key
    if st.sidebar.button("Update Visualization", help="Force update the visualization with current settings", key="update_viz_button"):
        # Create a spinner to show progress
        with st.spinner("Updating visualization..."):
            # Store the current value in session state for persistence
            st.session_state.max_accounts_display = max_accounts_display
            
            # Log the value for debugging
            st.sidebar.info(f"Updating visualization with max {max_accounts_display} accounts")
            
            # Force a complete data refresh by getting the full dataset from session state
            if 'network_data' in st.session_state and st.session_state.network_data:
                # Get the complete node set
                all_nodes, all_edges = st.session_state.network_data
                
                # Apply tweet data if available
                if 'all_nodes_with_tweets' in st.session_state:
                    for node_id, node_data in st.session_state.all_nodes_with_tweets.items():
                        if node_id in all_nodes and "tweet_summary" in node_data:
                            all_nodes[node_id]["tweet_summary"] = node_data["tweet_summary"]
                            all_nodes[node_id]["tweets"] = node_data.get("tweets", [])
                            all_nodes[node_id]["tweet_fetch_status"] = node_data.get("tweet_fetch_status", "")
                
                # Force the visualization to use the updated node set
                filtered_nodes = all_nodes
                edges = all_edges
    
        # Force a rerun to update the visualization with the new settings
        st.rerun()
    
    st.sidebar.header("Filter Criteria")
    
    # Numeric ranges with separate min/max inputs
    st.sidebar.subheader("Numeric Ranges")
    
    # Total Tweets Range
    st.sidebar.markdown("**Total Tweets Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tweets_min = st.number_input(
            "Min Tweets",
            min_value=0,
            max_value=1000000,
            value=0,
            step=1000,
            help="Minimum number of tweets an account must have"
        )
    with col2:
        tweets_max = st.number_input(
            "Max Tweets",
            min_value=0,
            max_value=1000000,
            value=1000000,
            step=1000,
            help="Maximum number of tweets an account can have"
        )
    total_tweets_range = (tweets_min, tweets_max)
    
    # Followers Range
    st.sidebar.markdown("**Followers Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        followers_min = st.number_input(
            "Min Followers",
            min_value=0,
            max_value=10000000,
            value=0,
            step=1000,
            help="Minimum number of followers an account must have"
        )
    with col2:
        followers_max = st.number_input(
            "Max Followers",
            min_value=0,
            max_value=10000000,
            value=10000000,
            step=1000,
            help="Maximum number of followers an account can have"
        )
    followers_range = (followers_min, followers_max)
    
    # Following Range
    st.sidebar.markdown("**Following Range**")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        following_min = st.number_input(
            "Min Following",
            min_value=0,
            max_value=10000000,
            value=0,
            step=1000,
            help="Minimum number of accounts this account must follow"
        )
    with col2:
        following_max = st.number_input(
            "Max Following",
            min_value=0,
            max_value=10000000,
            value=10000000,
            step=1000,
            help="Maximum number of accounts this account can follow"
        )
    following_range = (following_min, following_max)
    
    filters = {
        "statuses_range": total_tweets_range,
        "followers_range": followers_range,
        "friends_range": following_range,
        "media_range": (0, 10000),  # Default value since we removed the control
        "created_range": (datetime.date(2000, 1, 1), datetime.date(2100, 1, 1)),
        "require_location": False,
        "selected_locations": [],
        "require_blue_verified": False,
        "verified_option": "Any",
        "require_website": False,
        "business_account_option": "Any"
    }
    
    if 'network_data' not in st.session_state:
        st.session_state.network_data = None
    
    # Add these controls in the sidebar section where you have other controls
    st.sidebar.header("Data Fetch Options")

    following_pages = st.sidebar.number_input(
        "Pages of Following for Original Account",
        min_value=1,
        max_value=10,
        value=2,
        help="How many pages of following accounts to fetch for the original account (20 accounts per page)"
    )

    second_degree_pages = st.sidebar.number_input(
        "Pages of Following for Second Degree",
        min_value=1,
        max_value=5,
        value=1,
        help="How many pages of following accounts to fetch for each first-degree connection"
    )
    
    if st.button("Generate Network"):
        with st.spinner("Collecting network data..."):
            nodes, edges = run_async_main(
                input_username, 
                following_pages=following_pages,
                second_degree_pages=second_degree_pages
            )
            # Add debug information
            st.write(f"Retrieved {len(nodes)} nodes and {len(edges)} edges from API.")
            
            # Debug: Print first few nodes and edges
            st.write("Debug: First few nodes:", list(nodes.keys())[:5])
            st.write("Debug: First few edges:", edges[:5])
            
            # Always compute CloutRank on the full network, regardless of the UI setting
            with st.spinner("Computing network influence scores on full network..."):
                st.write("Debug: Starting PageRank computation on full network...")
                # Force CloutRank mode for the initial calculation
                st.session_state.importance_metric_mode = "CloutRank"
                full_cloutrank, incoming_contribs, outgoing_contribs = compute_cloutrank(nodes, edges, return_contributors=True)
                st.write(f"Debug: PageRank computation complete. Scores computed for {len(full_cloutrank)} nodes")
                st.write(f"Debug: Found {sum(len(c) for c in incoming_contribs.values())} total incoming contributions")
                st.write(f"Debug: Found {sum(len(c) for c in outgoing_contribs.values())} total outgoing contributions")
                
                # Store results in session state for later use
                st.session_state.full_cloutrank = full_cloutrank
                st.session_state.cloutrank_contributions = (incoming_contribs, outgoing_contribs)
                st.write("Debug: Stored PageRank results in session state")
        
        st.session_state.network_data = (nodes, edges)
        st.write("Debug: Stored network data in session state")
    
    if st.session_state.network_data is not None:
        nodes, edges = st.session_state.network_data
        filtered_nodes = filter_nodes(nodes, filters)
        
        # Add degree filtering
        st.sidebar.subheader("Node Degree Filtering")
        show_original = st.sidebar.checkbox(
            "Show Original Node", 
            value=True,
            help="Include the original account (the one you entered) in the visualization."
        )
        show_first_degree = st.sidebar.checkbox(
            "Show First Degree Connections", 
            value=True,
            help="Include accounts that are directly followed by the original account."
        )
        show_second_degree = st.sidebar.checkbox(
            "Show Second Degree Connections", 
            value=True,
            help="Include accounts that are followed by the first-degree connections."
        )
        
        # Filter nodes by degree
        original_id = next(id for id in filtered_nodes.keys() if id.startswith("orig_"))
        first_degree = {tgt for src, tgt in edges if src == original_id}
        second_degree = {tgt for src, tgt in edges if src in first_degree} - first_degree - {original_id}
        
        degree_filtered_nodes = {}
        for node_id, node in filtered_nodes.items():
            if node_id == original_id and show_original:
                degree_filtered_nodes[node_id] = node
            elif node_id in first_degree and show_first_degree:
                degree_filtered_nodes[node_id] = node
            elif node_id in second_degree and show_second_degree:
                degree_filtered_nodes[node_id] = node
        
        # Add community filtering
        if st.session_state.community_labels and st.session_state.community_colors:
            st.sidebar.subheader("Community Filtering")
            
            # Create a dictionary to store selected state of communities
            selected_communities = {}
            
            # Count accounts in each community
            community_counts = {}
            for username, comm_id in st.session_state.node_communities.items():
                if comm_id not in community_counts:
                    community_counts[comm_id] = 0
                community_counts[comm_id] += 1
            
            # Display checkboxes for each community with descriptive labels and account counts
            for comm_id, label in st.session_state.community_labels.items():
                count = community_counts.get(comm_id, 0)
                comm_label = f"{label} ({count} accounts)"
                selected_communities[comm_id] = st.sidebar.checkbox(
                    comm_label,
                    value=True,
                    key=f"community_{comm_id}"
                )
            
            # Filter nodes by selected communities
            community_filtered_nodes = {}
            for node_id, node in degree_filtered_nodes.items():
                if node_id.startswith("orig_"):
                    community_filtered_nodes[node_id] = node
                    continue
                    
                username = node["screen_name"]
                if username in st.session_state.node_communities:
                    community = st.session_state.node_communities[username]
                    if selected_communities.get(community, True):
                        community_filtered_nodes[node_id] = node
            
            filtered_nodes = community_filtered_nodes
        
        # Update node colors based on communities
        if st.session_state.community_labels and st.session_state.community_colors:
            for node_id in filtered_nodes:
                username = filtered_nodes[node_id]["screen_name"]
                if username in st.session_state.node_communities:
                    community = st.session_state.node_communities[username]
                    # Add safety check for community ID
                    if community in st.session_state.community_colors:
                        filtered_nodes[node_id]["community_color"] = st.session_state.community_colors[community]
                    else:
                        # Assign default color for unknown communities
                        st.warning(f"Community {community} not found in color mapping for user @{username}. Assigning to 'Other' category.")
                        filtered_nodes[node_id]["community_color"] = "#cccccc"  # Default gray color
                        # Reassign node to "Other" community if it exists
                        other_community = next((cid for cid, label in st.session_state.community_labels.items() 
                                             if label.lower() == "other"), "0")
                        st.session_state.node_communities[username] = other_community
        
        # Calculate importance scores based on filtering mode
        if use_pagerank:
            # Use the pre-computed scores from the full network
            importance_scores = st.session_state.full_cloutrank
        else:
            importance_scores = {
            node_id: sum(1 for _, tgt in edges if tgt == node_id)
            for node_id in filtered_nodes
        }

        # Also store the mode for reference in display functions
        st.session_state.importance_metric_mode = "CloutRank" if use_pagerank else "In-Degree"
        
        # Update size_factors dictionary
        size_factors = {
            'base_size': 1.0,  # Fixed value
            'importance_factor': float(account_size_factor),
            'label_size_factor': float(label_size)
        }
        
        # Display the graph
        if st.session_state.use_3d:
            # Use session state value if available
            display_max_accounts = st.session_state.get('max_accounts_display', max_accounts_display)
            
            # Show visual feedback on number of accounts being displayed
            st.info(f"Building 3D visualization with up to {display_max_accounts} accounts")
            st.write(f"DEBUG: max_accounts_display value = {display_max_accounts}, type = {type(display_max_accounts)}")
            
            html_code = build_network_3d(
                filtered_nodes, 
                edges,
                max_nodes=display_max_accounts,
                size_factors=size_factors,
                use_pagerank=use_pagerank
            )
            
            # Display node count information
            if 'last_selected_node_count' in st.session_state:
                st.caption(f"Selected {st.session_state.last_selected_node_count} nodes based on importance")
            
            components.html(html_code, height=750, width=800)
        else:
            # Use session state value if available
            display_max_accounts = st.session_state.get('max_accounts_display', max_accounts_display)
            
            # Show visual feedback on number of accounts being displayed
            st.info(f"Building 2D visualization with up to {display_max_accounts} accounts")
            
            net = build_network_2d(
                filtered_nodes, 
                edges,
                max_nodes=display_max_accounts,
                size_factors=size_factors,
                use_pagerank=use_pagerank
            )
            
            # Display node count information
            if 'last_selected_node_count_2d' in st.session_state:
                st.caption(f"Selected {st.session_state.last_selected_node_count_2d} nodes based on importance")
            
            net.save_graph("network.html")
            with open("network.html", 'r', encoding='utf-8') as f:
                components.html(f.read(), height=750, width=800)

        # Move community color key here, right after the graph
        if st.session_state.community_labels and st.session_state.community_colors:
            st.subheader("Community Color Key")
            
            # Count accounts in each community
            community_counts = {}
            for username, comm_id in st.session_state.node_communities.items():
                if comm_id not in community_counts:
                    community_counts[comm_id] = 0
                community_counts[comm_id] += 1
            
            # Convert community data to sorted list by labels with account counts
            community_data = []
            for comm_id, color in st.session_state.community_colors.items():
                label = st.session_state.community_labels.get(comm_id, f"Community {comm_id}")
                count = community_counts.get(comm_id, 0)
                label_with_count = f"{label} ({count} accounts)"
                community_data.append((label_with_count, color, comm_id))
            
            # Sort alphabetically by label
            community_data.sort(key=lambda x: x[0])
            
            # Calculate number of columns based on total communities
            num_communities = len(community_data)
            num_cols = min(4, max(2, 5 - (num_communities // 15)))
            
            # Calculate communities per column for even distribution
            communities_per_col = (num_communities + num_cols - 1) // num_cols if num_communities > 0 else 1
            
            # Create a container with fixed height and scrolling
            st.markdown("""
            <style>
            .community-grid {
                max-height: 400px;
                overflow-y: auto;
                padding-right: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a grid using Streamlit columns
            with st.container():
                st.markdown('<div class="community-grid">', unsafe_allow_html=True)
                
                # Create rows of communities instead of columns
                # This is more reliable than trying to create columns with HTML
                for i in range(0, num_communities, num_cols):
                    # Create a row of columns
                    cols = st.columns(num_cols)
                    
                    # Add communities to this row
                    for j in range(num_cols):
                        idx = i + j
                        if idx < num_communities:
                            label, color, _ = community_data[idx]
                            with cols[j]:
                                # Use a simple layout with colored text
                                st.markdown(
                                    f'<div style="display:flex; align-items:center">'
                                    f'<div style="width:15px; height:15px; background-color:{color}; '
                                    f'border-radius:3px; margin-right:8px;"></div>'
                                    f'<span style="font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{label}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                
                st.markdown('</div>', unsafe_allow_html=True)

        # Add community detection controls with tooltips
        st.header("Community Detection")
        col1, col2 = st.columns([2, 1])  # Change to two columns
        
        with col1:
            # Combined button for summarizing tweets and generating communities
            if st.button("Summarize Tweets & Generate Communities", use_container_width=True):
                # Add a state container to track progress
                state_container = st.empty()
                
                # Step 1: Summarize tweets
                with st.spinner("Step 1: Summarizing tweets for displayed accounts..."):
                    state_container.info("Summarizing tweets...")
                    
                    # Get the visualization nodes
                    original_id = next(id for id in filtered_nodes.keys() if id.startswith("orig_"))
                    followed_by_original = {tgt for src, tgt in edges if src == original_id}
                    
                    # Store importance scores in session state to ensure they're available for visualization updates
                    st.session_state.importance_scores = importance_scores
                    
                    # Select top nodes based on importance scores
                    top_overall = sorted(
                        [(nid, score) for nid, score in importance_scores.items() 
                         if not nid.startswith("orig_")],
                        key=lambda x: x[1],
                        reverse=True
                    )[:max_accounts_display // 2]
                    
                    top_independent = sorted(
                        [(nid, score) for nid, score in importance_scores.items()
                         if not nid.startswith("orig_") and nid not in followed_by_original
                         and nid not in {n for n, _ in top_overall}],  # Exclude accounts already in top_overall
                        key=lambda x: x[1],
                        reverse=True
                    )[:max_accounts_display // 2]
                    
                    # Create list of accounts to process
                    accounts_to_process = []
                    
                    # Debug the selection process
                    st.write(f"DEBUG: top_overall has {len(top_overall)} accounts")
                    st.write(f"DEBUG: top_independent has {len(top_independent)} accounts")
                    st.write(f"DEBUG: Merging these sets and removing duplicates...")
                    
                    # Combine the two lists but avoid duplicates by using a dictionary
                    combined_accounts = {}
                    for nid, score in top_overall:
                        combined_accounts[nid] = (nid, score, filtered_nodes[nid])
                    
                    for nid, score in top_independent:
                        if nid not in combined_accounts:  # Only add if not already in the dict
                            combined_accounts[nid] = (nid, score, filtered_nodes[nid])
                    
                    # Convert back to a list
                    accounts_to_process = list(combined_accounts.values())
                    
                    # Add original account if shown
                    if show_original:
                        accounts_to_process.append((original_id, 0, filtered_nodes[original_id]))
                    
                    # Double-check we have the right number of accounts to match visualization
                    st.write(f"DEBUG: Selected {len(accounts_to_process)} accounts for tweet summarization")
                    st.info(f"Summarizing tweets for {len(accounts_to_process)} displayed accounts...")
                    
                    # Process the accounts - this updates filtered_nodes with tweet summaries
                    asyncio.run(summarize_top_accounts(accounts_to_process, filtered_nodes, edges))
                    
                    # Ensure nodes variable is fully updated after tweet summarization
                    if st.session_state.network_data:
                        # Get updated nodes with tweet summaries
                        nodes, edges = st.session_state.network_data
                        # Update filtered_nodes with tweet data from nodes
                        for node_id in filtered_nodes:
                            if node_id in nodes and "tweet_summary" in nodes[node_id]:
                                filtered_nodes[node_id]["tweet_summary"] = nodes[node_id]["tweet_summary"]
                                filtered_nodes[node_id]["tweets"] = nodes[node_id].get("tweets", [])
                                filtered_nodes[node_id]["tweet_fetch_status"] = nodes[node_id].get("tweet_fetch_status", "")
                
                # Check if we have any tweet summaries
                has_tweet_data = any(
                    "tweet_summary" in node and node["tweet_summary"] 
                    for node_id, node in filtered_nodes.items()
                )
                
                if not has_tweet_data:
                    state_container.warning("No tweet summaries could be generated. Community generation may be less accurate.")
                else:
                    state_container.success(f"Successfully generated tweet summaries for some accounts.")
                
                # Step 2: Generate communities based on tweets
                with st.spinner("Step 2: Generating community labels from bios and tweets..."):
                    state_container.info("Generating community labels...")
                    
                    # Get the visualization nodes using FILTERED_NODES which now has tweet data
                    original_id = next(id for id in filtered_nodes.keys() if id.startswith("orig_"))
                    followed_by_original = {tgt for src, tgt in edges if src == original_id}
                    
                    # Select top nodes based on importance scores
                    top_overall = sorted(
                        [(nid, score) for nid, score in importance_scores.items() 
                         if not nid.startswith("orig_")],
                        key=lambda x: x[1],
                        reverse=True
                    )[:max_accounts_display // 2]
                    
                    top_independent = sorted(
                        [(nid, score) for nid, score in importance_scores.items()
                         if not nid.startswith("orig_") and nid not in followed_by_original
                         and nid not in {n for n, _ in top_overall}],  # Exclude accounts already in top_overall
                        key=lambda x: x[1],
                        reverse=True
                    )[:max_accounts_display // 2]
                    
                    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}
                    visualization_nodes = {node_id: meta for node_id, meta in filtered_nodes.items() 
                                        if node_id in selected_nodes}
                    
                    # Count how many accounts we're actually processing
                    displayed_account_count = len(visualization_nodes)
                    
                    # Always proceed with community generation if possible
                    if displayed_account_count < 3:  # Minimum needed for community detection
                        state_container.warning(f"Not enough accounts to form communities. Need at least 3 accounts.")
                    else:
                        # Create a container for progress tracking
                        progress_container = st.container()
                        with progress_container:
                            st.write("### Community Generation Progress")
                            # Display info about the process
                            st.info(f"Generating communities for {displayed_account_count} displayed accounts...")
                            
                            # Calculate appropriate number of communities based on dataset size
                            base_communities = 3  # Minimum number including "Other"
                            accounts_per_community = 12  # Target avg accounts per community
                            calculated_communities = base_communities + len(visualization_nodes) // accounts_per_community
                            num_communities = min(10, max(3, calculated_communities))
                            
                            # Generate community labels
                            community_labels = asyncio.run(generate_community_labels_with_tweets(
                                list(visualization_nodes.values()),
                                num_communities
                            ))
                            
                            # Debug the size of visualization_nodes 
                            st.write(f"DEBUG: Using {len(visualization_nodes)} accounts for community generation")
                            
                            if community_labels:
                                # Generate colors for communities
                                color_list = generate_n_colors(len(community_labels))
                                community_colors = {community_id: make_color_more_distinct(color) 
                                                 for community_id, color in zip(community_labels.keys(), color_list)}
                                
                                # Classify accounts into communities
                                node_communities = asyncio.run(classify_accounts_with_tweets(
                                    list(visualization_nodes.values()),
                                    community_labels
                                ))
                                
                                # Store results in session state
                                st.session_state.community_labels = community_labels
                                st.session_state.community_colors = community_colors
                                st.session_state.node_communities = node_communities
                                
                                # Store all nodes with tweet data in session state to ensure they're available for future visualizations
                                st.session_state.all_nodes_with_tweets = filtered_nodes
                                
                                # Set flag to automatically show tweet summaries in tables after successful generation
                                st.session_state.show_tweet_summaries = True
                                
                                # Update state container
                                state_container.success(f"""
                                    ✅ Process complete!
                                    - Generated tweet summaries 
                                    - Created {len(community_labels)} communities
                                    - Classified {len(node_communities)} accounts
                                """)
                                
                                # Force a rerun now that both steps are REALLY complete
                                st.rerun()
                            else:
                                state_container.error("Failed to generate community labels. Please try again.")

        with col2:
            st.info("Communities will be automatically determined based on account descriptions and tweets")

        # THEN Network Analysis (remove duplicated importance scores calculation)
        st.header("Network Analysis")
        # Add toggle for showing tweet summaries in tables
        has_tweet_data = any(
            "tweet_summary" in node and node["tweet_summary"] 
            for node_id, node in st.session_state.network_data[0].items()
        )
        
        # Use the session state flag if available, otherwise use has_tweet_data
        default_show_summaries = st.session_state.get('show_tweet_summaries', False) or has_tweet_data
        
        show_tweet_summaries = st.checkbox(
            "Show Tweet Summaries in Tables", 
            value=default_show_summaries,
            help="Include AI-generated summaries of tweets in the account tables"
        ) if has_tweet_data else False
        
        # Store the current preference back to session state
        st.session_state.show_tweet_summaries = show_tweet_summaries
        
        # Add toggle for excluding first-degree follows
        exclude_first_degree = st.checkbox(
            "Exclude First Degree Follows from Top Accounts", 
            value=False,
            help="When checked, accounts directly followed by the original account won't appear in the top accounts table."
        )
        
        # Modify display_top_accounts_table to include tweet summaries
        st.write(f"DEBUG: show_tweet_summaries = {show_tweet_summaries}")
        
        # Check if any nodes have tweet summaries
        has_summaries = any("tweet_summary" in node and node["tweet_summary"] for node_id, node in filtered_nodes.items())
        st.write(f"DEBUG: Nodes with tweet summaries: {has_summaries}")
        
        if has_summaries:
            # Show a sample of nodes with summaries
            summary_nodes = [(node_id, node["screen_name"], node.get("tweet_summary", "None")) 
                           for node_id, node in filtered_nodes.items() 
                           if "tweet_summary" in node and node["tweet_summary"]][:3]
            st.write(f"DEBUG: Sample summaries: {summary_nodes}")
        
        display_top_accounts_table(
            filtered_nodes,
            edges,
            importance_scores,
            original_id,
            exclude_first_degree,
            include_tweets=show_tweet_summaries
        )
        
        # Only show community tables if communities have been assigned
        if ('node_communities' in st.session_state and 
            st.session_state.node_communities and 
            'community_colors' in st.session_state and 
            st.session_state.community_colors):
            
            st.header("Community Analysis")
            # Get and display top accounts by community
            top_accounts = get_top_accounts_by_community(
                filtered_nodes,
                st.session_state.node_communities,
                importance_scores
            )
            
            # Update to include tweet summaries
            display_community_tables(
                top_accounts, 
                st.session_state.community_colors,
                filtered_nodes,  # Pass the filtered nodes
                edges,           # Pass the edges
                include_tweets=show_tweet_summaries
            )

    # Removed duplicate Update Visualization button to fix the StreamlitDuplicateElementId error

    # Add a download button for the full account data
    st.sidebar.markdown("---")  # Separator
    st.sidebar.header("Export Data")
    
    if st.session_state.network_data is not None:
        nodes, edges = st.session_state.network_data
        st.write("Debug: Creating downloadable table...")
        st.write(f"Debug: Using network data with {len(nodes)} nodes and {len(edges)} edges")
        st.write("Debug: PageRank mode:", st.session_state.get("importance_metric_mode"))
        st.write("Debug: Have full_cloutrank:", 'full_cloutrank' in st.session_state)
        st.write("Debug: Have contributions:", 'cloutrank_contributions' in st.session_state)
        
        df = create_downloadable_account_table(
            nodes, 
            edges, 
            include_tweets=('tweet_summaries' in st.session_state),
            include_communities=('node_communities' in st.session_state)
        )
        
        st.write(f"Debug: Created table with {len(df)} rows")
        st.write("Debug: Columns:", list(df.columns))
        st.write("Debug: Sample of contribution columns:")
        st.write(df[["Screen Name", "Incoming CloutRank Contributions", "Outgoing CloutRank Contributions"]].head())
        
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download All Account Data (CSV)",
            data=csv,
            file_name=f"{input_username}_account_data.csv",
            mime="text/csv",
        )
        
        # Option to view the full table in the app
        if st.sidebar.checkbox("Show Full Account Table"):
            st.header("Complete Account Data")
            st.dataframe(df, use_container_width=True)

async def get_user_tweets_async(user_id: str, session: aiohttp.ClientSession, cursor=None):
    """Asynchronously fetch tweets from a specific user"""
    endpoint = f"/UserTweets?user_id={user_id}"
    if cursor:
        endpoint += f"&cursor={cursor}"
    
    url = f"https://{RAPIDAPI_HOST}{endpoint}"
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": RAPIDAPI_HOST}
    
    try:
        async with session.get(url, headers=headers) as response:
            if response.status != 200:
                return [], None
            data = await response.text()
            return parse_tweet_data(data)
    except Exception as e:
        st.error(f"Error fetching tweets: {str(e)}")
        return [], None

def parse_tweet_data(json_str):
    """Parse the tweet data from the API response"""
    try:
        # First check if the response is valid JSON
        if not json_str or len(json_str.strip()) == 0:
            return [], None
            
        tweet_data = json.loads(json_str)
        tweets = []
        next_cursor = None
        
        # Basic validation of API response format
        if not isinstance(tweet_data, dict):
            return [], None
            
        # Check for error messages in the API response
        if "errors" in tweet_data:
            error_msgs = [error.get("message", "Unknown error") for error in tweet_data.get("errors", [])]
            if error_msgs:
                st.error(f"API returned errors: {', '.join(error_msgs)}")
            return [], None
            
        # Handle case where API returns empty data
        if "data" not in tweet_data:
            return [], None
            
        # Navigate to the timeline instructions
        timeline = tweet_data.get("data", {}).get("user_result_by_rest_id", {}).get("result", {}).get("profile_timeline_v2", {}).get("timeline", {})
        
        # Check for suspended or protected accounts
        if not timeline:
            user_result = tweet_data.get("data", {}).get("user_result_by_rest_id", {}).get("result", {})
            if user_result:
                if user_result.get("__typename") == "UserUnavailable":
                    reason = user_result.get("reason", "Account unavailable")
                    return [], None
            return [], None
        
        # Get all entries from the timeline
        all_entries = []
        
        # First check for pinned entry
        for instruction in timeline.get("instructions", []):
            if instruction.get("__typename") == "TimelinePinEntry":
                entry = instruction.get("entry", {})
                if entry:
                    all_entries.append(entry)
            
            # Then get regular entries
            elif instruction.get("__typename") == "TimelineAddEntries":
                entries = instruction.get("entries", [])
                for entry in entries:
                    # Skip cursor entries
                    if entry.get("content", {}).get("__typename") == "TimelineTimelineCursor":
                        cursor_type = entry.get("content", {}).get("cursor_type")
                        if cursor_type == "Bottom":
                            next_cursor = entry.get("content", {}).get("value")
                        continue
                    all_entries.append(entry)
        
        # Process each entry
        for entry in all_entries:
            tweet_content = entry.get("content", {}).get("content", {}).get("tweet_results", {}).get("result", {})
            
            # Skip if no tweet content
            if not tweet_content:
                continue
            
            # Check if it's a retweet
            is_retweet = False
            tweet_to_parse = tweet_content
            
            # For retweets, get the original tweet
            if "retweeted_status_results" in tweet_content.get("legacy", {}):
                is_retweet = True
                tweet_to_parse = tweet_content.get("legacy", {}).get("retweeted_status_results", {}).get("result", {})
            
            # Extract tweet data
            legacy = tweet_to_parse.get("legacy", {})
            
            # Skip if no legacy data
            if not legacy:
                continue
            
            # Get tweet text
            text = legacy.get("full_text", "")
            
            # Clean URLs from text
            urls = legacy.get("entities", {}).get("urls", [])
            for url in urls:
                text = text.replace(url.get("url", ""), url.get("display_url", ""))
            
            # Get engagement metrics
            likes = legacy.get("favorite_count", 0)
            retweets = legacy.get("retweet_count", 0)
            replies = legacy.get("reply_count", 0)
            quotes = legacy.get("quote_count", 0)
            
            # Get tweet date
            created_at = legacy.get("created_at", "")
            try:
                date_obj = dt.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                date_str = date_obj.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = created_at
            
            # Add to tweets list
            tweets.append({
                "text": text,
                "date": date_str,
                "likes": likes,
                "retweets": retweets,
                "replies": replies,
                "quotes": quotes,
                "total_engagement": likes + retweets + replies + quotes,
                "is_retweet": is_retweet
            })
    
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON response from API: {str(e)}")
        # Try to log a sample of the response for debugging
        if json_str:
            sample = json_str[:100] + "..." if len(json_str) > 100 else json_str
            st.error(f"Response sample: {sample}")
        return [], None
    except Exception as e:
        st.error(f"Error parsing tweet data: {str(e)}")
        return [], None
    
    return tweets, next_cursor

async def generate_tweet_summary(tweets, username):
    """Generate an AI summary of tweets using Gemini API"""
    if not tweets:
        return "No tweets available"
    
    # Prepare tweet texts for the prompt
    tweet_texts = [f"- {tweet['date']}: {tweet['text']}" for tweet in tweets[:20]]  # Limit to 20 tweets
    tweet_content = "\n".join(tweet_texts)
    
    prompt = f"""Analyze these recent tweets from @{username} and provide a brief summary (max 100 words) of their main topics, interests, and tone:

{tweet_content}

Summary:"""

    try:
        # Initialize Gemini client
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel("gemini-2.0-flash")
        
        # Generate summary
        response = client.generate_content(prompt)
        summary = response.text.strip()
        return summary
    
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return "Error generating summary"

def chunk_accounts_for_processing(accounts, max_chunk_size=400):
    """Split accounts into manageable chunks for API processing."""
    chunks = []
    current_chunk = []
    
    for account in accounts:
        current_chunk.append(account)
        if len(current_chunk) >= max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
    
    # Add any remaining accounts
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def merge_community_labels(label_groups):
    """
    Merge multiple sets of community labels, resolving duplicates
    and maintaining consistent naming.
    """
    if not label_groups:
        return {}
    
    # If only one group, just return it
    if len(label_groups) == 1:
        return label_groups[0]
    
    # Start with the first group
    merged_labels = label_groups[0].copy()
    
    # Track label frequencies to identify the most common themes
    label_counts = {}
    for labels in label_groups:
        for community, label in labels.items():
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
    
    # Sort labels by frequency (most common first)
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create a comprehensive set of labels that covers all communities
    next_community_id = max([int(c) for c in merged_labels.keys()]) + 1 if merged_labels else 0
    
    # Add all labels from subsequent groups
    for group_idx, labels in enumerate(label_groups[1:], 1):
        for community, label in labels.items():
            # If this exact label is already in merged_labels, skip
            if label in merged_labels.values():
                continue
                
            # Otherwise add as a new community
            merged_labels[str(next_community_id)] = label
            next_community_id += 1
    
    return merged_labels

async def generate_community_labels_with_tweets(accounts, num_communities):
    """
    Generate community labels using both account descriptions and tweet summaries.
    Handles large account sets by chunking and merging results.
    """
    # Prepare accounts with tweet summaries
    accounts_with_tweets = []
    for account in accounts:
        tweet_summary = account.get("tweet_summary", "")
        # Only include accounts that have either a description or tweet summary
        if account.get("description") or tweet_summary:
            account_info = {
                "screen_name": account.get("screen_name", ""),
                "description": account.get("description", ""),
                "tweet_summary": tweet_summary
            }
            accounts_with_tweets.append(account_info)
    
    if not accounts_with_tweets:
        return {}
    
    # Check if we need to chunk (roughly estimate token count)
    estimated_tokens = sum(len(a["description"].split()) + len(a.get("tweet_summary", "").split()) 
                          for a in accounts_with_tweets)
    need_chunking = estimated_tokens > 20000  # Conservative limit for GPT-4o-mini
    
    if need_chunking:
        st.info(f"Processing {len(accounts_with_tweets)} accounts in chunks for better results")
        
        # Chunk accounts
        chunks = chunk_accounts_for_processing(accounts_with_tweets)
        
        # Process each chunk
        label_groups = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            st.write(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} accounts)...")
            chunk_labels = await process_account_chunk_with_tweets(chunk, num_communities)
            label_groups.append(chunk_labels)
            progress_bar.progress((i+1)/len(chunks))
        
        # Merge results
        return merge_community_labels(label_groups)
    else:
        # Process all accounts at once
        return await process_account_chunk_with_tweets(accounts_with_tweets, num_communities)

async def process_account_chunk_with_tweets(accounts, num_communities):
    """Process accounts to generate community labels using tweet data."""
    # Format account data for the prompt
    accounts_text = []
    for i, account in enumerate(accounts):
        screen_name = account["screen_name"]
        description = account["description"]
        tweet_summary = account.get("tweet_summary", "")
        
        account_text = f"Account {i+1} (@{screen_name}):\n"
        account_text += f"Bio: {description}\n"
        if tweet_summary:
            account_text += f"Tweet Summary: {tweet_summary}\n"
        accounts_text.append(account_text)
    
    accounts_formatted = "\n".join(accounts_text)
    
    # If text is very long, summarize
    prompt_text = accounts_formatted
    if len(accounts_formatted) > 1500000000:
        # Calculate how many accounts we're truncating
        truncated_text = accounts_formatted[:1500000000]
        # Count how many accounts are in the truncated text by looking for Account markers
        accounts_in_truncated = truncated_text.count("Account ")
        remaining_accounts = len(accounts) - accounts_in_truncated
        # Add a note about the truncated accounts
        prompt_text = truncated_text + f"\n\n[... and {remaining_accounts} more accounts with similar patterns ...]"
    
    prompt = f"""I'm analyzing a Twitter network and need to group accounts into communities based on their characteristics. 
    
For each account, I'll provide their Twitter bio description and a summary of their recent tweets.

Please analyze these accounts and create community labels that effectively group similar accounts together. Include an "Other" category for accounts that don't fit well into specific groups.

Accounts:
{prompt_text}

Return your response as a JSON object mapping community IDs to descriptive labels (1-3 words each). For example:
{{
  "0": "Tech Leaders",
  "1": "News Media",
  "2": "Other"
}}"""

    try:
        # Initialize Gemini client
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel("gemini-2.0-flash")
        
        # Generate community labels
        response = client.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON part from the response
        try:
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                community_labels = json.loads(json_str)
                return community_labels
            else:
                st.warning(f"Could not extract JSON from the API response")
                return {}
        except json.JSONDecodeError:
            st.warning(f"Invalid JSON format in the API response")
            return {}
            
    except Exception as e:
        st.error(f"Error generating community labels: {str(e)}")
        return {}

async def classify_accounts_with_tweets(accounts, community_labels):
    """
    Classify accounts into communities based on their descriptions and tweet summaries.
    Returns a dictionary mapping account screen_names to community IDs.
    """
    # Extract all accounts with either a description or tweet summary
    accounts_to_classify = []
    for account in accounts:
        if account.get("description") or account.get("tweet_summary"):
            accounts_to_classify.append({
                "screen_name": account.get("screen_name"),
                "description": account.get("description", ""),
                "tweet_summary": account.get("tweet_summary", "")
            })
    
    if not accounts_to_classify:
        return {}
    
    # Get valid community IDs
    valid_community_ids = set(community_labels.keys())
    
    # Format community labels for the prompt
    community_desc = "\n".join([f"- Community {cid}: {label}" for cid, label in community_labels.items()])
    
    # Find the "Other" category ID
    other_community_id = next((cid for cid, label in community_labels.items() 
                            if label.lower() == "other"), list(community_labels.keys())[0])
    
    # Process in chunks - CHANGED from 100 to 50
    max_accounts_per_call = 50
    chunks = [accounts_to_classify[i:i+max_accounts_per_call] 
              for i in range(0, len(accounts_to_classify), max_accounts_per_call)]
    
    # Process each chunk
    results = {}
    progress_bar = st.progress(0)
    
    for i, chunk in enumerate(chunks):
        # Create prompt for this chunk
        chunk_texts = []
        for account in chunk:
            screen_name = account["screen_name"]
            description = account["description"]
            tweet_summary = account.get("tweet_summary", "")
            
            account_text = f"Account (@{screen_name}):\n"
            account_text += f"Bio: {description}\n"
            if tweet_summary:
                account_text += f"Tweet Summary: {tweet_summary}\n"
            chunk_texts.append(account_text)
        
        chunk_formatted = "\n".join(chunk_texts)
        
        prompt = f"""I have a set of Twitter accounts that I need to classify into predefined communities.
        
The communities are:
{community_desc}

Please classify each of the following accounts into one of these communities based on their bio and tweet content.
Be thorough in your analysis of each account's content and assign it to the most appropriate community.

VERY IMPORTANT: 
1. You MUST use ONLY the COMMUNITY ID NUMBERS provided above (e.g., "0", "1", "2") - not the label names
2. DO NOT create new community IDs - only use the exact numeric IDs listed above
3. The "Other" category (ID: {other_community_id}) should be used for accounts that don't clearly fit into any specific community

Accounts to classify:
{chunk_formatted}

Provide your answer as a JSON object mapping the Twitter username (without @) to the community ID (as a string). For example:
{{
  "username1": "0",
  "username2": "1", 
  "username3": "{other_community_id}"
}}
        
IMPORTANT: Each username must be assigned to exactly one of these community IDs: {', '.join(valid_community_ids)}
"""
        # Call API to classify accounts
        try:
            # Initialize Gemini client
            genai.configure(api_key=GEMINI_API_KEY)
            client = genai.GenerativeModel("gemini-2.0-flash")
            
            # Generate classifications
            response = client.generate_content(prompt)
            
            # Parse response
            response_text = response.text.strip()
            
            try:
                import re
                json_match = re.search(r'({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1)
                    chunk_results = json.loads(json_str)
                    
                    # Validate and clean results
                    for username, community in chunk_results.items():
                        if username.startswith('@'):
                            username = username[1:]  # Remove @ if present
                        
                        # Special case for the "Other" category
                        if community.lower() == "other" or community == "Other":
                            # Find the community ID for the "Other" label
                            other_community = next((cid for cid, label in community_labels.items() 
                                                if label.lower() == "other"), None)
                            
                            if other_community:
                                results[username] = other_community
                                continue
                        
                        # Normal case - ensure community ID is valid
                        if community in valid_community_ids:
                            results[username] = community
                        else:
                            # If community ID is invalid, assign to "Other" category if it exists
                            other_community = next((cid for cid, label in community_labels.items() 
                                                if label.lower() == "other"), list(community_labels.keys())[0])
                            results[username] = other_community
                            st.warning(f"Invalid community ID {community} assigned to @{username}. Reassigning to {community_labels[other_community]}.")
                else:
                    st.warning(f"Could not extract JSON from chunk {i+1} response")
            except json.JSONDecodeError:
                st.warning(f"Invalid JSON format in chunk {i+1} response")
        except Exception as e:
            st.error(f"Error classifying accounts in chunk {i+1}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i+1)/len(chunks))
    
    return results

# First, let's fix the visualization coloring function to use proper labels
def get_node_colors(nodes, node_communities, community_colors):
    """
    Assigns colors to nodes based on their community.
    Returns a dictionary mapping node_id -> color.
    """
    node_colors = {}
    for node_id, node in nodes.items():
        # Get the node's username
        username = node.get("screen_name", "")
        
        # Determine the community and color
        if username in node_communities:
            community_id = node_communities[username]
            # Get the descriptive label for this community ID
            node_colors[node_id] = community_colors.get(community_id, "#cccccc")
        else:
            # Default color for nodes without a community
            node_colors[node_id] = "#cccccc"
    
    return node_colors

def display_tweet_fetch_summary(nodes, processed_accounts):
    """Display a summary of tweet fetch statuses to help diagnose patterns in API failures"""
    # Count different types of errors
    status_counts = {}
    accounts_by_status = {}
    
    for node_id, node in nodes.items():
        if "tweet_fetch_status" in node:
            status = node["tweet_fetch_status"]
            # Simplify the status for categorization
            if not status:
                status = "Success (tweets available)"
            
            # Count occurrences of each status
            if status not in status_counts:
                status_counts[status] = 0
                accounts_by_status[status] = []
            
            status_counts[status] += 1
            accounts_by_status[status].append(node["screen_name"])
    
    # If no fetch statuses, return early
    if not status_counts:
        return
    
    st.write("### Tweet Fetch Status Summary")
    
    # Display counts in a table
    status_data = {
        "Status": [],
        "Count": [],
        "Percentage": []
    }
    
    total = sum(status_counts.values())
    
    for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
        status_data["Status"].append(status)
        status_data["Count"].append(count)
        status_data["Percentage"].append(f"{(count/total)*100:.1f}%")
    
    st.table(status_data)
    
    # Show detailed breakdown for each status type with examples
    st.write("### Detailed Status Breakdown")
    for status, accounts in accounts_by_status.items():
        with st.expander(f"{status} ({len(accounts)} accounts)"):
            # Show up to 10 example accounts
            examples = accounts[:10]
            st.write(", ".join([f"@{username}" for username in examples]))
            if len(accounts) > 10:
                st.write(f"...and {len(accounts) - 10} more")

async def summarize_top_accounts(top_accounts, nodes, edges):
    """Process tweet summarization with adaptive concurrency and batch processing"""
    st.write(f"Starting tweet summarization process for {len(top_accounts)} accounts...")
    
    # Start with a conservative limit that we know works
    concurrency_limit = 10
    batch_size = 5  # Process 5 accounts in a single API call
    
    # Create a session-level variable to track failures
    if 'fd_failures' not in st.session_state:
        st.session_state.fd_failures = 0
    
    # If we've had failures, reduce the limit
    if st.session_state.fd_failures > 0:
        concurrency_limit = max(5, concurrency_limit - st.session_state.fd_failures)
        st.warning(f"Reducing concurrency to {concurrency_limit} due to previous failures")
    
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    # Configure connection with limits
    conn = aiohttp.TCPConnector(limit=concurrency_limit, force_close=True, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    
    # Counter for failures within this run
    failures_this_run = 0
    
    # Fetch tweets for a batch of accounts
    async def fetch_tweets_for_batch(account_batch):
        nonlocal failures_this_run  # Move nonlocal declaration to the beginning of the function
        
        tweet_data = []
        
        for node_id, _, node in account_batch:
            username = node["screen_name"]
            try:
                tweets, _ = await get_user_tweets_async(node_id, session, cursor=None)
                if not tweets:
                    # Add error metadata to help diagnose why no tweets were returned
                    node["tweet_fetch_status"] = "No tweets returned by API"
                else:
                    node["tweet_fetch_status"] = ""  # Clear status on success
                tweet_data.append((username, tweets))
            except aiohttp.ClientResponseError as e:
                failures_this_run += 1
                error_msg = f"API response error (status {e.status}): {str(e)}"
                st.error(f"Error fetching tweets for @{username}: {error_msg}")
                node["tweet_fetch_status"] = error_msg
                tweet_data.append((username, []))
            # ... other exception handlers ...
                
        return tweet_data
    
    # Process a batch of accounts with semaphore
    async def process_account_batch(batch_idx, account_batch):
        nonlocal failures_this_run  # Also add nonlocal declaration here
        
        async with semaphore:
            try:
                # Fetch tweets for this batch
                tweet_data = await fetch_tweets_for_batch(account_batch)
                
                # Process results
                results = []
                for i, (username, tweets) in enumerate(tweet_data):
                    # Get the node_id and node object for this account
                    node_id, _, node = account_batch[i]
                    
                    if tweets:
                        # Generate summary for tweets
                        summary = await generate_tweet_summary(tweets, username)
                        
                        # Store summary and tweets directly in the node object
                        node["tweet_summary"] = summary
                        node["tweets"] = tweets
                        
                        # Also update the node in the main nodes dictionary
                        nodes[node_id]["tweet_summary"] = summary
                        nodes[node_id]["tweets"] = tweets
                        
                        results.append((username, True, summary))
                    else:
                        # Set empty summary in the node
                        node["tweet_summary"] = "No tweets available"
                        nodes[node_id]["tweet_summary"] = "No tweets available"
                        
                        results.append((username, False, "No tweets available"))
                        
                return batch_idx, results
                
            except Exception as e:
                failures_this_run += 1
                st.error(f"Error processing batch {batch_idx}: {str(e)}")
                return batch_idx, [(node["screen_name"], False, f"Error: {str(e)}") for _, _, node in account_batch]
    
    try:
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            progress_bar = st.progress(0)
            status_text = st.empty()
            concurrency_info = st.empty()
            concurrency_info.info(f"Using concurrency level: {concurrency_limit} with batch size: {batch_size}")
            
            # Divide accounts into batches of batch_size
            batches = []
            for i in range(0, len(top_accounts), batch_size):
                batches.append((i // batch_size, top_accounts[i:i+batch_size]))
            
            # Debug output to confirm batch count
            st.write(f"Created {len(batches)} batches from {len(top_accounts)} accounts")
            
            # Debug total number of accounts
            total_in_batches = sum(len(batch) for _, batch in batches)
            st.write(f"DEBUG: Total accounts in all batches: {total_in_batches}")
            
            # Create tasks for all batches
            tasks = [process_account_batch(batch_idx, account_batch) for batch_idx, account_batch in batches]
            
            # Show real-time progress as results arrive
            successful_accounts = 0
            total_accounts = len(top_accounts)
            
            for future in asyncio.as_completed(tasks):
                batch_idx, results = await future
                
                # Count successful accounts in this batch
                batch_successful = sum(1 for _, success, _ in results if success)
                successful_accounts += batch_successful
                
                # Also update the nodes with summaries from the results
                for username, success, summary in results:
                    # Find the node with this username
                    for node_id, node_data in nodes.items():
                        if node_data.get('screen_name') == username:
                            # Make sure the summary is properly set
                            if success:
                                # The summary should already be set in the process_account_batch function,
                                # but let's make sure it's there
                                if "tweet_summary" not in node_data or not node_data["tweet_summary"]:
                                    node_data["tweet_summary"] = summary
                            else:
                                # Set a default message for failures
                                node_data["tweet_summary"] = "No tweet summary available"
                                
                            # Mark that we've processed this node
                            node_data["tweet_summary_processed"] = True
                            break
                
                # Show completion message
                st.write(f"✅ Processed batch {batch_idx+1}/{len(batches)} ({batch_successful}/{len(results)} accounts successful)")
                
                # Update progress
                progress = (batch_idx + 1) / len(batches)
                progress_bar.progress(progress)
                status_text.text(f"Processed {successful_accounts}/{total_accounts} accounts")
            
            # Complete progress
            progress_bar.progress(1.0)
            
            # Show completion message
            status_text.text(f"Tweet summarization complete! Successful: {successful_accounts}/{total_accounts}")
            
            # Display error summary to help diagnose patterns
            display_tweet_fetch_summary(nodes, top_accounts)
            
            # Update session state
            st.session_state.network_data = (nodes, edges)
            
            # Also store filtered_nodes specifically to ensure they have tweet summaries
            if 'all_nodes_with_tweets' not in st.session_state:
                st.session_state.all_nodes_with_tweets = {}
                
            # Update all_nodes_with_tweets with any nodes that have tweet summaries
            for node_id, node_data in nodes.items():
                if "tweet_summary" in node_data and node_data["tweet_summary"]:
                    if node_id not in st.session_state.all_nodes_with_tweets:
                        st.session_state.all_nodes_with_tweets[node_id] = {}
                    
                    # Copy only the necessary fields
                    st.session_state.all_nodes_with_tweets[node_id] = node_data.copy()
            
            # Enable tweet summaries checkbox by default
            st.session_state.show_tweet_summaries = True
            
            # Don't force a rerun here - let the combined function handle it
            # st.rerun()
    
    except Exception as e:
        st.error(f"Error in tweet summarization: {str(e)}")
        st.session_state.fd_failures += 1  # Increment failure count on overall errors
    finally:
        # Ensure connector is closed
        await conn.close()

async def generate_batch_tweet_summaries(batch_data, batch_size=5):
    """Generate summaries for multiple accounts (batch_size) in a single API call"""
    if not batch_data:
        return {}
    
    # Prepare batch content for the prompt
    batch_content = []
    usernames = []
    account_tweets_map = {}  # Track number of tweets per account
    
    # Format each account's tweets
    for username, tweets in batch_data:
        usernames.append(username)
        account_tweets_map[username] = len(tweets)
        
        # Check if there are any tweets for this account
        if not tweets:
            continue
            
        account_tweets = tweets[:10]  # Limit to 10 tweets per account for brevity
        tweet_texts = [f"@{username} {tweet['date']}: {tweet['text']}" for tweet in account_tweets]
        batch_content.extend(tweet_texts)
        # Add separator between accounts
        if len(batch_content) > 0:
            batch_content.append("---")  # Separator between accounts
    
    # Remove the last separator
    if batch_content and batch_content[-1] == "---":
        batch_content.pop()
    
    # Initialize results with specific error messages for accounts with no tweets
    summaries = {}
    for username in usernames:
        if account_tweets_map[username] == 0:
            summaries[username] = "No tweets available for this account"
    
    # If no accounts have tweets, return early
    if not batch_content:
        return summaries
    
    prompt = f"""Analyze the following tweets from {len([u for u in usernames if account_tweets_map[u] > 0])} different Twitter accounts.
For EACH account, provide a brief summary (max 50 words) of their main topics, interests, and tone.

Tweets:
{chr(10).join(batch_content)}

Response format:
@username1: [brief summary of account 1]
@username2: [brief summary of account 2]
...and so on for all accounts
"""

    try:
        # Initialize Gemini client
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel("gemini-2.0-flash")
        
        # Generate batch summaries
        response = client.generate_content(prompt)
        
        # Parse the response to extract summaries for each account
        text = response.text.strip()
        
        for line in text.split('\n'):
            if line.startswith('@') and ':' in line:
                username, summary = line.split(':', 1)
                username = username.strip('@').strip()
                if username in usernames:
                    summaries[username] = summary.strip()
        
        # Ensure all accounts with tweets have summaries
        for username in usernames:
            if username not in summaries and account_tweets_map[username] > 0:
                summaries[username] = "API generated no summary despite tweets being available"
        
        return summaries
    
    except Exception as e:
        error_message = str(e)
        st.error(f"Error generating batch summaries: {error_message}")
        
        # Provide specific error messages based on the exception
        error_type = "API error"
        if "rate limit" in error_message.lower():
            error_type = "Rate limit exceeded"
        elif "timeout" in error_message.lower():
            error_type = "API timeout"
        elif "auth" in error_message.lower() or "key" in error_message.lower():
            error_type = "API authentication error"
        
        # Set error message for all accounts that don't already have summaries
        for username in usernames:
            if username not in summaries:
                summaries[username] = f"Error: {error_type} - {error_message[:50]}..."
        
        return summaries

def make_color_more_distinct(hex_color):
    """Make colors more distinct by increasing saturation and adjusting value"""
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # Increase saturation, ensure good value
    s = min(1.0, s * 1.3)  # Increase saturation by 30%
    v = max(0.6, min(0.95, v))  # Keep value in a good range
    
    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
    # Convert to hex
    return '#{:02x}{:02x}{:02x}'.format(
        int(r * 255),
        int(g * 255),
        int(b * 255))

def categorize_communities(community_labels):
    """Group communities into broader categories for better organization"""
    # Define category keywords
    categories = {
        "Technology": ["ai", "software", "dev", "tech", "computer", "engineer", "automation", "robotics", 
                      "open source", "security", "privacy", "uiux", "design"],
        "Business": ["vc", "investor", "startup", "founder", "ceo", "business", "marketing", 
                    "growth", "sales", "careers", "jobs", "y combinator"],
        "Finance": ["financial", "trading", "crypto", "web3"],
        "Politics": ["political", "government", "regulation", "regulatory", "conflict", "ukraine", 
                    "russia", "israel", "palestine"],
        "Science": ["research", "academic", "neuroscience", "bci", "science", "stem", "space", 
                   "exploration", "health", "longevity", "theoretical"],
        "Creative": ["artist", "designer", "music", "arts", "food"],
        "Social": ["community", "support", "personal", "sports", "culture", "family", "e/acc", "reflection"],
        "Media": ["news", "journalism", "publisher", "book"],
        "Geographic": ["indian", "chinese", "irish"],
        "Other": ["other", "random"]
    }
    
    # Create mapping from community ID to category
    community_categories = {}
    
    for comm_id, label in community_labels.items():
        assigned = False
        label_lower = label.lower()
        
        # Try to find matching category
        for category, keywords in categories.items():
            if any(keyword in label_lower for keyword in keywords):
                community_categories[comm_id] = category
                assigned = True
                break
        
        # If no category matches, use "Other"
        if not assigned:
            community_categories[comm_id] = "Other"
    
    return community_categories

if __name__ == "__main__":
    main()
