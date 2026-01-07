from huggingface_hub import HfApi, login
import shutil
from pathlib import Path

# Login to HuggingFace
print("Please login to Hugging Face...")
login()

api = HfApi()
user_info = api.whoami()
username = user_info['name']

print(f"\n✓ Logged in as: {username}")

# Your space ID
space_id = f"{username}/protein-sst-tracking"
print(f"\n📦 Uploading to: {space_id}")

# Files to upload
files_to_upload = [
    ("app.py", "The Gradio dashboard app"),
    ("requirements.txt", "Python dependencies"),
    (".trackio", "TrackIO database folder")
]

print("\n" + "=" * 60)
print("Uploading files...")
print("=" * 60)

for file_path, description in files_to_upload:
    path = Path(file_path)
    if path.exists():
        print(f"\n📤 Uploading {file_path} - {description}")
        try:
            if path.is_dir():
                # Upload folder
                api.upload_folder(
                    folder_path=str(path),
                    repo_id=space_id,
                    repo_type="space",
                    path_in_repo=path.name
                )
            else:
                # Upload single file
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=path.name,
                    repo_id=space_id,
                    repo_type="space",
                )
            print(f"   ✓ Uploaded successfully")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
    else:
        print(f"\n⚠️ Skipping {file_path} - not found")

print("\n" + "=" * 60)
print("🎉 Upload complete!")
print("=" * 60)
print(f"\n🌐 Your dashboard URL:")
print(f"   https://huggingface.co/spaces/{space_id}")
print(f"\n⏳ Note: It may take 1-2 minutes for the Space to build and start.")
print(f"   Refresh the page if you see setup instructions.")
