"""
Simple GitHub Repository Cleanup
Clean up your GitHub profile to focus on professional brain-inspired AI research
"""

import requests
import json

def get_user_repositories(username, token):
    """Get all user repositories"""
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    repos = []
    page = 1
    
    while True:
        url = f"https://api.github.com/users/{username}/repos"
        params = {
            "type": "owner",
            "sort": "updated",
            "direction": "desc",
            "page": page,
            "per_page": 100
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break
        
        page_repos = response.json()
        if not page_repos:
            break
            
        repos.extend(page_repos)
        page += 1
        
        if page > 10:
            break
    
    return repos

def analyze_repositories(repos):
    """Analyze repositories and categorize them"""
    
    keep_repos = []
    delete_repos = []
    private_repos = []
    
    # Repositories to keep
    keep_patterns = [
        "brain-inspired-ai",
        "brain-ai", 
        "neural",
        "neuroscience",
        "computational",
        "research",
        "leo267410-dev"
    ]
    
    # Repositories to delete
    delete_patterns = [
        "experiment",
        "test",
        "demo",
        "practice",
        "learning",
        "tutorial",
        "temp",
        "backup",
        "old",
        "unused",
        "playground",
        "sandbox"
    ]
    
    for repo in repos:
        repo_name = repo["name"].lower()
        
        # Always keep profile repository
        if repo_name == "leo267410-dev":
            keep_repos.append(repo)
            continue
        
        # Always keep brain-inspired-ai
        if "brain-inspired-ai" in repo_name:
            keep_repos.append(repo)
            continue
        
        # Check if private
        if repo["private"]:
            private_repos.append(repo)
            continue
        
        # Decision logic
        should_keep = any(pattern in repo_name for pattern in keep_patterns)
        should_delete = any(pattern in repo_name for pattern in delete_patterns)
        
        if should_keep and not should_delete:
            keep_repos.append(repo)
        elif should_delete and not should_keep:
            delete_repos.append(repo)
        elif not should_keep and not should_delete:
            delete_repos.append(repo)
        else:
            keep_repos.append(repo)
    
    return {
        "keep": keep_repos,
        "delete": delete_repos,
        "private": private_repos
    }

def generate_cleanup_plan(analysis):
    """Generate cleanup plan"""
    
    plan = "# GitHub Repository Cleanup Plan\n\n"
    plan += f"## Analysis Results\n\n"
    plan += f"### Repositories to KEEP ({len(analysis['keep'])})\n\n"
    
    for repo in analysis['keep']:
        plan += f"- **{repo['name']}** - {repo.get('description', 'No description')}\n"
        plan += f"  - Stars: {repo['stargazers_count']}, Forks: {repo['forks_count']}\n\n"
    
    plan += f"### Repositories to DELETE ({len(analysis['delete'])})\n\n"
    
    for repo in analysis['delete']:
        plan += f"- **{repo['name']}** - {repo.get('description', 'No description')}\n"
        plan += f"  - Stars: {repo['stargazers_count']}, Forks: {repo['forks_count']}\n"
        plan += f"  - **Reason**: Doesn't support professional AI researcher identity\n\n"
    
    if analysis['private']:
        plan += f"### Private Repositories ({len(analysis['private'])})\n\n"
        for repo in analysis['private']:
            plan += f"- **{repo['name']}** - {repo.get('description', 'No description')}\n\n"
    
    plan += """
## Cleanup Strategy

### Recommended Actions:
1. **Archive** repositories with some activity (safer than delete)
2. **Delete** repositories with 0 stars and 0 forks
3. **Keep** only repositories that support AI/neuroscience research

### Manual Deletion Steps:
For each repository to delete:
1. Go to: https://github.com/leo267410-dev/REPO-NAME/settings
2. Scroll to bottom
3. Click "Delete this repository"
4. Confirm deletion

### Archive Instead (Recommended):
1. Go to repository settings
2. Scroll to "Danger Zone"  
3. Click "Archive this repository"

## Benefits:
- Professional focus on brain-inspired AI research
- Clear messaging to visitors
- Better search ranking with focused topics
- Attracts AI/neuroscience opportunities
"""
    
    return plan

def generate_deletion_commands(repos):
    """Generate deletion commands"""
    
    commands = "#!/bin/bash\n\n"
    commands += "# GitHub Repository Cleanup Commands\n"
    commands += "# WARNING: This will permanently delete repositories!\n\n"
    commands += "echo 'Repositories to be deleted:'\n"
    
    for repo in repos:
        commands += f"echo '- {repo['name']} ({repo['stargazers_count']} stars)'\n"
    
    commands += "\necho 'To delete these repositories, visit:'\n\n"
    
    for repo in repos:
        commands += f"# Delete {repo['name']}\n"
        commands += f"# Visit: https://github.com/leo267410-dev/{repo['name']}/settings\n\n"
    
    commands += "# Or archive instead:\n"
    for repo in repos:
        commands += f"# Archive {repo['name']}\n"
        commands += f"# Visit: https://github.com/leo267410-dev/{repo['name']}/settings\n\n"
    
    return commands

def main():
    """Main cleanup function"""
    
    username = "leo267410-dev"
    token = "YOUR_GITHUB_TOKEN_HERE"  # Replace with your token
    
    print("Analyzing GitHub repositories...")
    
    # Get repositories
    repos = get_user_repositories(username, token)
    print(f"Found {len(repos)} repositories")
    
    # Analyze
    analysis = analyze_repositories(repos)
    
    print(f"Repositories to keep: {len(analysis['keep'])}")
    print(f"Repositories to delete: {len(analysis['delete'])}")
    print(f"Private repositories: {len(analysis['private'])}")
    
    # Save materials
    import os
    os.makedirs("cleanup_materials", exist_ok=True)
    
    # Save cleanup plan
    with open("cleanup_materials/cleanup_plan.md", "w") as f:
        f.write(generate_cleanup_plan(analysis))
    
    # Save deletion commands
    with open("cleanup_materials/deletion_commands.sh", "w") as f:
        f.write(generate_deletion_commands(analysis['delete']))
    
    # Save analysis
    with open("cleanup_materials/analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print("\nCleanup materials generated!")
    print("Files created:")
    print("- cleanup_materials/cleanup_plan.md")
    print("- cleanup_materials/deletion_commands.sh") 
    print("- cleanup_materials/analysis.json")
    
    return analysis

if __name__ == "__main__":
    main()
