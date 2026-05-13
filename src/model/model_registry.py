# model_registry.py - Full Model Lifecycle Management
import json
from mlflow.tracking import MlflowClient
import mlflow
import dagshub
from datetime import datetime
import sys

dagshub.init(repo_owner='email4prasanth', repo_name='mlflow_ci', mlflow=True)
mlflow.set_experiment("Final_modle")

# Load run ID and model name from json
reports_path = "reports/run_info.json"
with open(reports_path, "r") as file:
    run_info = json.load(file)

run_id = run_info['run_id']
model_name = run_info['model_name']

# Create mlflow client
client = MlflowClient()

def get_latest_version():
    """Get the latest version of the model"""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise Exception(f"No versions found for model '{model_name}'")
    return max([int(v.version) for v in versions])

def get_version_by_alias(alias):
    """Get model version by alias"""
    try:
        model_details = client.get_registered_model(model_name)
        return model_details.aliases.get(alias)
    except:
        return None

def move_to_staging(version=None):
    """Move a model version to Staging"""
    if version is None:
        version = get_latest_version()
    
    # Set staging alias
    client.set_registered_model_alias(
        name=model_name,
        alias="Staging",
        version=version
    )
    
    # Add staging metadata
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="staging_date",
        value=datetime.now().isoformat()
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="current_stage",
        value="staging"
    )
    
    print(f"✅ Model {model_name} version {version} moved to Staging")
    print(f"   📍 URI: models:/{model_name}@Staging")
    return version

def move_to_production(version=None):
    """Move a model version to Production"""
    if version is None:
        # If version not specified, get current staging version
        staging_version = get_version_by_alias("Staging")
        if staging_version:
            version = staging_version
            print(f"   Promoting Staging version {version} to Production")
        else:
            version = get_latest_version()
            print(f"   No Staging version found. Promoting latest version {version} to Production")
    
    # Set production alias
    client.set_registered_model_alias(
        name=model_name,
        alias="Production",
        version=version
    )
    
    # Add production metadata
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="production_date",
        value=datetime.now().isoformat()
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="current_stage",
        value="production"
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="deployed_by",
        value="mlops_pipeline"
    )
    
    print(f"✅ Model {model_name} version {version} moved to Production")
    print(f"   📍 URI: models:/{model_name}@Production")
    return version

def move_to_archived(version=None):
    """Move a model version to Archived"""
    if version is None:
        # Get current production version
        prod_version = get_version_by_alias("Production")
        if prod_version:
            version = prod_version
            print(f"   Archiving Production version {version}")
        else:
            # If no production version, get staging
            staging_version = get_version_by_alias("Staging")
            if staging_version:
                version = staging_version
                print(f"   No Production version. Archiving Staging version {version}")
            else:
                version = get_latest_version()
                print(f"   No Production or Staging version. Archiving latest version {version}")
    
    # Set archived alias
    client.set_registered_model_alias(
        name=model_name,
        alias="Archived",
        version=version
    )
    
    # Remove from production and staging if present
    try:
        if get_version_by_alias("Production") == version:
            client.delete_registered_model_alias(name=model_name, alias="Production")
            print("   Removed from Production")
    except:
        pass
    
    try:
        if get_version_by_alias("Staging") == version:
            client.delete_registered_model_alias(name=model_name, alias="Staging")
            print("   Removed from Staging")
    except:
        pass
    
    # Add archived metadata
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="archived_date",
        value=datetime.now().isoformat()
    )
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="current_stage",
        value="archived"
    )
    
    print(f"✅ Model {model_name} version {version} moved to Archived")
    print(f"   📍 URI: models:/{model_name}@Archived")
    return version

def rollback_to_version(version):
    """Rollback to a specific version"""
    # Move specified version to production
    move_to_production(version)
    
    # Add rollback tag
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="rollback_date",
        value=datetime.now().isoformat()
    )
    
    print(f"⚠️  Rolled back to version {version} in Production")

def list_model_versions():
    """List all versions and their current aliases"""
    versions = client.search_model_versions(f"name='{model_name}'")
    model_details = client.get_registered_model(model_name)
    
    print(f"\n{'='*60}")
    print(f"📊 Model: {model_name}")
    print(f"{'='*60}")
    
    # Show current aliases
    print(f"\n🏷️  Current Aliases:")
    for alias, ver in model_details.aliases.items():
        print(f"   @{alias} → version {ver}")
    
    # Show all versions
    print(f"\n📦 All Versions:")
    for version in sorted(versions, key=lambda v: int(v.version), reverse=True):
        # Check which aliases point to this version
        aliases = []
        for alias, ver in model_details.aliases.items():
            if str(ver) == version.version:
                aliases.append(f"@{alias}")
        
        alias_str = f" [{', '.join(aliases)}]" if aliases else ""
        
        # Get stage from tags
        stage = version.tags.get("current_stage", "none")
        
        print(f"   Version {version.version}{alias_str}")
        print(f"     Status: {version.status}, Stage: {stage}")
        if version.tags:
            print(f"     Tags: {dict(list(version.tags.items())[:3])}")  # Show first 3 tags
    
    return versions

def show_model_uri_examples():
    """Show examples of how to use model URIs"""
    print(f"\n{'='*60}")
    print(f"📌 Model URI Examples for Loading:")
    print(f"{'='*60}")
    print(f"   Latest Staging:  models:/{model_name}@Staging")
    print(f"   Production:      models:/{model_name}@Production")
    print(f"   Archived:        models:/{model_name}@Archived")
    print(f"   Specific version: models:/{model_name}/1")
    print(f"\n   Load in code:")
    print(f"   model = mlflow.sklearn.load_model('models:/{model_name}@Production')")

# Main execution with command-line interface
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n❌ Please specify an action!")
        print("\n📖 Usage:")
        print("   python model_registry.py [ACTION]")
        print("\n🎯 Actions:")
        print("   staging     - Move latest model to Staging")
        print("   production  - Move Staging model to Production")
        print("   archive     - Archive Production model")
        print("   promote     - Shortcut: staging → production → archive")
        print("   list        - List all versions and aliases")
        print("   rollback    - Rollback Production to previous version")
        print("   help        - Show this help message")
        print("\n📝 Examples:")
        print("   python model_registry.py staging")
        print("   python model_registry.py production")
        print("   python model_registry.py archive")
        print("   python model_registry.py list")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    try:
        if action == "staging":
            print(f"\n🚀 Moving model to Staging...")
            move_to_staging()
            list_model_versions()
            show_model_uri_examples()
            
        elif action == "production":
            print(f"\n🚀 Promoting model to Production...")
            move_to_production()
            list_model_versions()
            show_model_uri_examples()
            
        elif action == "archive":
            print(f"\n📦 Archiving model...")
            move_to_archived()
            list_model_versions()
            
        elif action == "promote":
            print(f"\n🔄 Full promotion cycle (Staging → Production → Archive)...")
            move_to_staging()
            move_to_production()
            move_to_archived()
            list_model_versions()
            
        elif action == "list":
            list_model_versions()
            show_model_uri_examples()
            
        elif action == "rollback":
            if len(sys.argv) > 2:
                version = int(sys.argv[2])
                print(f"\n⚠️  Rolling back to version {version}...")
                rollback_to_version(version)
                list_model_versions()
            else:
                print("\n❌ Please specify version number!")
                print("   Example: python model_registry.py rollback 3")
                
        elif action == "help":
            print("\n📖 Model Registry Lifecycle Management")
            print("="*60)
            print("\nWorkflow:")
            print("   1. Model is logged → Available in registry")
            print("   2. python model_registry.py staging    → Ready for testing")
            print("   3. python model_registry.py production → Deployed to production")
            print("   4. python model_registry.py archive    → Deprecated/archived")
            print("\nYou can also load models by alias in your code:")
            print("   - models:/Best Model@Staging")
            print("   - models:/Best Model@Production")
            print("   - models:/Best Model@Archived")
            
        else:
            print(f"\n❌ Unknown action: {action}")
            print("   Run 'python model_registry.py help' for available actions")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)