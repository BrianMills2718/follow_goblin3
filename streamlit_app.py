#!/usr/bin/env python3
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
#  2. Node importance calculated using either PageRank or In-Degree
#  3. Configurable node and label sizes
#  4. Display filters for:
#     - statuses_count, followers_count, friends_count, media_count
#     - created_at date range
#     - location (with search)
#     - verification status
#     - website presence
#     - business account status
#  5. Paginated tables showing:
#     - Top accounts by importance (PageRank/In-Degree)
#     - Top independent accounts (not followed by original account)


import streamlit as st
import streamlit.components.v1 as components
import asyncio
import aiohttp
import json
from pyvis.network import Network
import datetime
import numpy as np
from scipy import sparse
from openai import OpenAI
from typing import List, Dict
import colorsys
import random
from tqdm import tqdm  # For progress tracking
import os

# Set page to wide mode - this must be the first Streamlit command
st.set_page_config(layout="wide", page_title="X Network Analysis", page_icon="🔍")

# CONSTANTS
RAPIDAPI_KEY = st.secrets["RAPIDAPI_KEY"]
RAPIDAPI_HOST = "twitter283.p.rapidapi.com"  # Updated API host

# Add these constants after the existing RAPIDAPI constants
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
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

def compute_pagerank(nodes, edges, damping=0.85, epsilon=1e-8, max_iter=100):
    """
    Compute PageRank for each node in the network.
    """
    # Create node index mapping
    node_to_index = {node_id: idx for idx, node_id in enumerate(nodes.keys())}
    n = len(nodes)
    
    # Create adjacency matrix
    rows, cols = [], []
    for src, tgt in edges:
        if src in node_to_index and tgt in node_to_index:
            rows.append(node_to_index[src])
            cols.append(node_to_index[tgt])
    
    # Create sparse matrix
    data = np.ones_like(rows)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Normalize adjacency matrix
    out_degree = np.array(A.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1  # Avoid division by zero
    A = sparse.diags(1/out_degree) @ A
    
    # Initialize PageRank
    pr = np.ones(n) / n
    
    # Power iteration
    for _ in range(max_iter):
        pr_next = (1 - damping) / n + damping * A.T @ pr
        if np.sum(np.abs(pr_next - pr)) < epsilon:
            break
        pr = pr_next
    
    # Convert back to dictionary
    return {node_id: pr[idx] for node_id, idx in node_to_index.items()}

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
    """Initialize OpenAI client."""
    return OpenAI(api_key=OPENAI_API_KEY)

async def get_community_labels(accounts: List[Dict], num_communities: int) -> List[str]:
    """Get community labels from GPT-4o-mini."""
    client = get_openai_client()
    
    # Create prompt with account information
    account_info = "\n".join([
        f"Username: {acc['screen_name']}, Description: {acc['description']}"
        for acc in accounts[:20]  # Use first 20 accounts as examples
    ])
    
    prompt = f"""Based on these X/Twitter accounts and their descriptions:

{account_info}

Generate exactly {num_communities} distinct community labels that best categorize these and similar accounts.
Include an "Other" category as one of the {num_communities} labels. Return only the labels, one per line.
Focus on professional/topical communities rather than demographic categories."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that categorizes social media accounts into communities."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    
    # Parse response into list of labels
    labels = [label.strip() for label in response.choices[0].message.content.split('\n') if label.strip()]
    return labels

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
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that classifies social media accounts into predefined communities. Only use the exact community labels provided."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                )
                
                # Parse response into dictionary
                classifications = {}
                for line in response.choices[0].message.content.split('\n'):
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
    # Set default size factors if None
    if size_factors is None:
        size_factors = {
            'base_size': 5,
            'importance_factor': 3.0,
            'label_size_factor': 1.0
        }

    # Determine node importance
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
            
    pagerank = compute_pagerank(nodes, edges)
    importance = pagerank if use_pagerank else in_degrees

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

    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}

    # Filter nodes and edges to only include selected nodes
    nodes = {node_id: meta for node_id, meta in nodes.items() if node_id in selected_nodes}
    edges = [(src, tgt) for src, tgt in edges if src in selected_nodes and tgt in selected_nodes]

    nodes_data = []
    links_data = []

    # Convert edges to proper format
    links_data = [{"source": str(src), "target": str(tgt)} for src, tgt in edges]

    # Convert nodes to proper format with additional info
    for node_id, meta in nodes.items():
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
            
            nodes_data.append({
                "id": str(node_id),
                "name": str(meta.get("screen_name", "")),
                "community": community,
                "followers": int(followers_count),
                "following": int(following_count),
                "ratio": float(ratio),
                "size": float(node_size),
                "description": str(meta.get("description", "")),
                "pagerank": float(importance.get(node_id, 0)),
                "indegree": int(in_degrees.get(node_id, 0)),
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
                "pagerank": 0.0,
                "indegree": 0,
                "color": "#6ca6cd"
            })

    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links_data)

    html_code = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <script src="https://unpkg.com/three@0.149.0/build/three.min.js"></script>
        <script src="https://unpkg.com/3d-force-graph@1.70.10/dist/3d-force-graph.min.js"></script>
        <script src="https://unpkg.com/three-spritetext"></script>
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
                    PageRank: ${{node.pagerank.toFixed(4)}}<br/>
                    In-Degree: ${{node.indegree}}
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
    # Use same importance calculation as 3D version
    in_degrees = {node_id: 0 for node_id in nodes.keys()}
    for src, tgt in edges:
        if tgt in in_degrees:
            in_degrees[tgt] += 1
    
    pagerank = compute_pagerank(nodes, edges)
    importance = pagerank if use_pagerank else in_degrees
    
    # Find original node and followed nodes
    original_id = next(id for id in nodes.keys() if id.startswith("orig_"))
    followed_by_original = {tgt for src, tgt in edges if src == original_id}
    
    # Select nodes same way as 3D version
    top_overall = sorted(
        [(nid, score) for nid, score in importance.items() 
         if not nid.startswith("orig_")],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]
    
    top_independent = sorted(
        [(nid, score) for nid, score in importance.items() 
         if not nid.startswith("orig_") and nid not in followed_by_original],
        key=lambda x: x[1],
        reverse=True
    )[:max_nodes//2]
    
    selected_nodes = {original_id} | {nid for nid, _ in top_overall} | {nid for nid, _ in top_independent}

    # Create pyvis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    # Normalize importance scores
    max_importance = max(importance.values())
    normalized_importance = {nid: score/max_importance for nid, score in importance.items()}
    
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
        title = (f"Followers: {followers_str}\n"
                 f"Following: {friends_str}\n"
                 f"Ratio: {ratio_str}\n"
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

def display_community_tables(top_accounts_by_community, colors, include_tweets=False):
    """Display tables for top accounts in each community."""
    
    for community_id, accounts in top_accounts_by_community.items():
        # Skip if no accounts
        if not accounts:
            continue
        
        # Get community color and label
        color = colors.get(community_id, "#cccccc")
        
        # CHANGE: Use descriptive label instead of just "Community X"
        if 'community_labels' in st.session_state and st.session_state.community_labels:
            label = st.session_state.community_labels.get(community_id, f"Community {community_id}")
        else:
            label = f"Community {community_id}"
        
        # Style the community header with its color
        st.markdown(
            f"<h3 style='color:{color}'>{label}</h3>", 
            unsafe_allow_html=True
        )
        
        # Create table data
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
        
        # Only show top 10 per community
        show_accounts = accounts[:10]
        
        # Populate table
        for idx, (node_id, node, score) in enumerate(show_accounts, 1):
            table_data["Rank"].append(idx)
            table_data["Username"].append(node["screen_name"])
            table_data["Score"].append(f"{score:.4f}")
            table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
            table_data["Following"].append(f"{node.get('friends_count', 0):,}")
            table_data["Description"].append(node.get("description", ""))
        
            # Add tweet summary if requested
            if include_tweets:
                table_data["Tweet Summary"].append(node.get("tweet_summary", "No summary available"))
        
        # Display table
        st.table(table_data)

def display_top_accounts_table(nodes, edges, importance_scores, original_id, exclude_first_degree=False, include_tweets=False):
    """Display table of top accounts based on importance scores."""
    st.subheader(f"Top {min(20, len(nodes)-1)} Accounts Overall")
    
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
    
    # Create table data
    table_data = {
        "Rank": [],
        "Username": [],
        "Score": [],
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
        
        # Only add community data if communities exist
        if communities_exist:
            # Get community label if available
            community_id = communities.get(node["screen_name"], "0")
            community_label = "N/A"
            if community_id in community_labels:
                community_label = community_labels[community_id]
            else:
                community_label = f"Community {community_id}"
            table_data["Community"].append(community_label)
        
        table_data["Score"].append(f"{score:.4f}")
        table_data["Followers"].append(f"{node.get('followers_count', 0):,}")
        table_data["Following"].append(f"{node.get('friends_count', 0):,}")
        table_data["Description"].append(node.get("description", ""))
    
        # Add tweet summary if requested
        if include_tweets:
            table_data["Tweet Summary"].append(node.get("tweet_summary", "No summary available"))
    
    # Display table
    st.table(table_data)

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
        options=["In-Degree", "PageRank"],
        index=0,
        help="In-Degree measures importance by how many accounts follow this account in the network. PageRank considers both quantity and quality of connections."
    )
    use_pagerank = (importance_metric == "PageRank")
    
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
        value=50,
        step=5,
        help="Maximum number of accounts to show in the visualization. Lower values improve performance."
    )
    
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
        
        st.session_state.network_data = (nodes, edges)
    
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
            
            # Display checkboxes for each community with descriptive labels
            for comm_id, label in st.session_state.community_labels.items():
                comm_label = f"{label} (Community {comm_id})"
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
        # Update size_factors dictionary
        size_factors = {
            'base_size': 1.0,  # Fixed value
            'importance_factor': float(account_size_factor),
            'label_size_factor': float(label_size)
        }
        
        # Display the graph
        if st.session_state.use_3d:
            html_code = build_network_3d(
                filtered_nodes, 
                edges,
                max_nodes=max_accounts_display,
                size_factors=size_factors,
                use_pagerank=use_pagerank
            )
            st.write("Debug: About to render 3D graph")
            components.html(html_code, height=750, width=800)
            st.write("Debug: Finished rendering")
        else:
            net = build_network_2d(
                filtered_nodes, 
                edges,
                max_nodes=max_accounts_display,
                size_factors=size_factors,
                use_pagerank=use_pagerank
            )
            net.save_graph("network.html")
            with open("network.html", 'r', encoding='utf-8') as f:
                components.html(f.read(), height=750, width=800)

        # Move community color key here, right after the graph
        if st.session_state.community_labels and st.session_state.community_colors:
            st.subheader("Community Color Key")
            cols = st.columns(len(st.session_state.community_colors))
            for idx, (comm_id, color) in enumerate(st.session_state.community_colors.items()):
                with cols[idx]:
                    # Get descriptive label instead of just community ID
                    label = st.session_state.community_labels.get(comm_id, f"Community {comm_id}")
                    st.markdown(
                        f'<div style="background-color: {color}; padding: 10px; '
                        f'border-radius: 5px; margin: 2px 0; color: black; '
                        f'text-align: center;">{label}</div>',
                        unsafe_allow_html=True
                    )

        # Add community detection controls with tooltips
        st.header("Community Detection")
        col1, col2, col3 = st.columns(3)  # Add a third column

        with col1:
            num_communities = st.number_input(
                "Number of Communities (including 'Other')",
                min_value=2,
                max_value=15,
                value=min(5, len(filtered_nodes)),
                help="How many distinct communities to identify in the network"
            )

        with col2:
            # Keep the existing "Generate Community Labels" button (bio-only)
            if st.button("Generate Community Labels (Bio Only)"):
                with st.spinner("Generating community labels from account descriptions..."):
                    # Check if we have enough accounts
                    if len(filtered_nodes) < num_communities:
                        st.warning(f"Not enough accounts to form {num_communities} communities. Reduce the number of communities.")
                    else:
                        # Generate community labels
                        community_labels = asyncio.run(generate_community_labels(
                            list(filtered_nodes.values()),
                            num_communities
                        ))
                        
                        # Generate community colors as a dictionary mapping community IDs to colors
                        color_list = generate_n_colors(num_communities)
                        community_colors = {community_id: color for community_id, color in 
                                           zip(community_labels.keys(), color_list)}
                        
                        # Store results in session state
                        st.session_state.community_colors = community_colors  # Now it's a dictionary
                        st.session_state.community_labels = community_labels
                        
                        # Force a rerun to update the visualization
                        st.rerun()

        with col3:
            # Add the new button for tweet-enhanced labels
            if st.button("Generate Labels with Tweets"):
                with st.spinner("Generating community labels from bios and tweets..."):
                    # Check if we have tweet summaries
                    has_tweet_data = any(
                        "tweet_summary" in node and node["tweet_summary"] 
                        for node_id, node in filtered_nodes.items()
                    )
                    
                    if not has_tweet_data:
                        st.warning("No tweet summaries found. Please summarize tweets first.")
                    elif len(filtered_nodes) < num_communities:
                        st.warning(f"Not enough accounts to form {num_communities} communities. Reduce the number of communities.")
                    else:
                        # Generate community labels using tweet data - only use the filtered nodes for defining communities
                        community_labels = asyncio.run(generate_community_labels_with_tweets(
                            list(filtered_nodes.values()),
                            num_communities
                        ))
                        
                        # Generate community colors as a dictionary
                        color_list = generate_n_colors(num_communities)
                        community_colors = {community_id: color for community_id, color in 
                                          zip(community_labels.keys(), color_list)}
                        
                        # Only classify the filtered/displayed nodes
                        node_communities = asyncio.run(classify_accounts_with_tweets(
                            list(filtered_nodes.values()),
                            community_labels
                        ))
                        
                        # Store results in session state
                        st.session_state.community_colors = community_colors
                        st.session_state.community_labels = community_labels
                        st.session_state.node_communities = node_communities
                        
                        # Force a rerun to update the visualization
                        st.rerun()

        # Calculate importance scores first so they're available for both sections
        importance_scores = compute_pagerank(filtered_nodes, edges) if use_pagerank else {
            node_id: sum(1 for _, tgt in edges if tgt == node_id)
            for node_id in filtered_nodes
        }

        # AFTER Community Detection and importance scores, THEN Tweet Analysis
        st.header("Tweet Analysis")
        col1, col2 = st.columns(2)

        # Use the max_accounts_display as the number to summarize
        with col1:
            st.info(f"Will summarize tweets for top {max_accounts_display} accounts")

        with col2:
            if st.button("Summarize Tweets for Top Accounts"):
                with st.spinner("Fetching tweets and generating summaries..."):
                    # Get top accounts based on importance scores
                    top_accounts = []
                    for node_id, node in filtered_nodes.items():
                        if not node_id.startswith("orig_"):
                            score = importance_scores.get(node_id, 0)
                            top_accounts.append((node_id, score, node))
                    
                    # Sort by score and limit to same number as visualization
                    top_accounts.sort(key=lambda x: x[1], reverse=True)
                    top_accounts = top_accounts[:max_accounts_display]
                    
                    # Run the async function with proper parameters
                    asyncio.run(summarize_top_accounts(top_accounts, nodes, edges))

        # THEN Network Analysis (remove duplicated importance scores calculation)
        st.header("Network Analysis")
        
        # Add toggle for showing tweet summaries in tables
        has_tweet_data = any(
            "tweet_summary" in node and node["tweet_summary"] 
            for node_id, node in st.session_state.network_data[0].items()
        )
        
        show_tweet_summaries = st.checkbox(
            "Show Tweet Summaries in Tables", 
            value=has_tweet_data,
            help="Include AI-generated summaries of tweets in the account tables"
        ) if has_tweet_data else False
        
        # Add toggle for excluding first-degree follows
        exclude_first_degree = st.checkbox(
            "Exclude First Degree Follows from Top Accounts", 
            value=False,
            help="When checked, accounts directly followed by the original account won't appear in the top accounts table."
        )
        
        # Modify display_top_accounts_table to include tweet summaries
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
                include_tweets=show_tweet_summaries
            )

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
        tweet_data = json.loads(json_str)
        tweets = []
        next_cursor = None
        
        # Navigate to the timeline instructions
        timeline = tweet_data.get("data", {}).get("user_result_by_rest_id", {}).get("result", {}).get("profile_timeline_v2", {}).get("timeline", {})
        
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
                date_obj = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
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
    
    except Exception as e:
        st.error(f"Error parsing tweet data: {str(e)}")
        return [], None
    
    return tweets, next_cursor

async def generate_tweet_summary(tweets, username):
    """Generate an AI summary of tweets using OpenAI API"""
    if not tweets:
        return "No tweets available"
    
    # Prepare tweet texts for the prompt
    tweet_texts = [f"- {tweet['date']}: {tweet['text']}" for tweet in tweets[:20]]  # Limit to 20 tweets
    tweet_content = "\n".join(tweet_texts)
    
    prompt = f"""Analyze these recent tweets from @{username} and provide a brief summary (max 100 words) of their main topics, interests, and tone:

{tweet_content}

Summary:"""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes Twitter content concisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
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
    """Process a chunk of accounts to generate community labels using tweet data."""
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
    
    # Create the prompt
    prompt = f"""I'm analyzing a Twitter network and need to group accounts into {num_communities} distinct communities based on their characteristics. 
    
For each account, I'll provide their Twitter bio description and a summary of their recent tweets.

Please analyze the following accounts and suggest {num_communities} community labels that best categorize these accounts. 
Consider both their bio descriptions AND their tweet content.

IMPORTANT: One of your community labels MUST be named exactly "Other" for accounts that don't fit well into specific categories.

Accounts:
{accounts_formatted}

Please provide exactly {num_communities} descriptive community labels, each 1-3 words. Format your response as a JSON object with community IDs (0 to {num_communities-1}) as keys and labels as values. For example:
{{
  "0": "Tech Enthusiasts",
  "1": "Political Commentators",
  "{num_communities-1}": "Other"
}}
"""

    # Call OpenAI API to generate community labels
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes Twitter networks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content.strip()
        
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
        except json.JSONDecodeError:
            st.warning(f"Invalid JSON format in the API response")
            
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
1. You MUST use the COMMUNITY ID (the number) not the label name in your response
2. The "Other" category (ID: {other_community_id}) should be used for accounts that don't clearly fit into any specific community

Accounts to classify:
{chunk_formatted}

Provide your answer as a JSON object mapping the Twitter username (without @) to the community ID (as a string). For example:
{{
  "username1": "{list(community_labels.keys())[0]}",
  "username2": "{list(community_labels.keys())[1] if len(community_labels) > 1 else list(community_labels.keys())[0]}",
  "username3": "{other_community_id}"
}}
"""
        
        # Call OpenAI API to classify accounts
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes Twitter accounts. Only use the exact community IDs provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
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

async def summarize_top_accounts(top_accounts, nodes, edges):
    """Process tweet summarization with adaptive concurrency and batch processing"""
    st.write("Starting tweet summarization process...")
    
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
        tweet_data = []
        
        for node_id, _, node in account_batch:
            username = node["screen_name"]
            try:
                tweets, _ = await get_user_tweets_async(node_id, session, cursor=None)
                tweet_data.append((username, tweets))
            except Exception as e:
                nonlocal failures_this_run
                failures_this_run += 1
                st.error(f"Error fetching tweets for @{username}: {str(e)}")
                tweet_data.append((username, []))
                
        return tweet_data
    
    # Process a batch of accounts with semaphore
    async def process_account_batch(batch_idx, account_batch):
        async with semaphore:
            try:
                # Fetch tweets for all accounts in the batch
                tweet_data = await fetch_tweets_for_batch(account_batch)
                
                # Generate summaries for the batch in a single API call
                summaries = await generate_batch_tweet_summaries(tweet_data, batch_size)
                
                # Update the nodes with summaries
                results = []
                for i, (node_id, _, node) in enumerate(account_batch):
                    username = node["screen_name"]
                    tweets = tweet_data[i][1] if i < len(tweet_data) else []
                    summary = summaries.get(username, "No summary available")
                    
                    # Update the node
                    nodes[node_id]["tweets"] = tweets
                    nodes[node_id]["tweet_summary"] = summary
                    
                    # Add to results
                    results.append((username, len(tweets) > 0, "Success"))
                
                return batch_idx, results
                
            except Exception as e:
                nonlocal failures_this_run
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
            
            # Update session state
            st.session_state.network_data = (nodes, edges)
            
            # Force a rerun to update the visualization
            st.rerun()
    
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
    
    # Format each account's tweets
    for username, tweets in batch_data:
        usernames.append(username)
        account_tweets = tweets[:10]  # Limit to 10 tweets per account for brevity
        tweet_texts = [f"@{username} {tweet['date']}: {tweet['text']}" for tweet in account_tweets]
        batch_content.extend(tweet_texts)
        # Add separator between accounts
        if len(batch_content) > 0:
            batch_content.append("---")  # Separator between accounts
    
    # Remove the last separator
    if batch_content and batch_content[-1] == "---":
        batch_content.pop()
    
    prompt = f"""Analyze the following tweets from {len(usernames)} different Twitter accounts.
For EACH account, provide a brief summary (max 50 words) of their main topics, interests, and tone.

Tweets:
{chr(10).join(batch_content)}

Response format:
@username1: [brief summary of account 1]
@username2: [brief summary of account 2]
...and so on for all accounts
"""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes Twitter content. Provide concise summaries for multiple accounts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=len(usernames) * 100,  # Allow enough tokens for all summaries
            temperature=0.7
        )
        
        # Parse the response to extract summaries for each account
        summaries = {}
        text = response.choices[0].message.content.strip()
        
        for line in text.split('\n'):
            if line.startswith('@') and ':' in line:
                username, summary = line.split(':', 1)
                username = username.strip('@').strip()
                if username in usernames:
                    summaries[username] = summary.strip()
        
        # Ensure all accounts have summaries
        for username in usernames:
            if username not in summaries:
                summaries[username] = "No summary generated"
        
        return summaries
    
    except Exception as e:
        st.error(f"Error generating batch summaries: {str(e)}")
        return {username: f"Error: {str(e)}" for username in usernames}

if __name__ == "__main__":
    main()
