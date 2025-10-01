import os
import urllib.request
import zipfile
from pathlib import Path
import ssl
import time

# Bypass SSL verification if needed (for some corporate networks)
ssl._create_default_https_context = ssl._create_unverified_context


def download_file_with_progress(url, filename):
    """Download file with progress indicator"""
    if os.path.exists(filename):
        print(f"✓ {filename} already exists")
        return True

    print(f"Downloading {filename}...")

    try:
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="", flush=True)

        urllib.request.urlretrieve(url, filename, report_progress)
        print(f"\n✓ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n✗ Failed to download {filename}: {e}")
        return False


def main():
    # Create directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("gfpgan/weights").mkdir(parents=True, exist_ok=True)

    print("🚀 Starting SadTalker Model Download...")
    print("=" * 60)

    # Essential models (uncomment the ones you need)
    essential_models = [
        # These are the files mentioned in your error - UNCOMMENT THESE:
        ("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/auido2exp_00300-model.pth",
         "checkpoints/auido2exp_00300-model.pth"),
        ("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/auido2pose_00140-model.pth",
         "checkpoints/auido2pose_00140-model.pth"),
        ("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/epoch_20.pth", "checkpoints/epoch_20.pth"),
        ("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/facevid2vid_00189-model.pth.tar",
         "checkpoints/facevid2vid_00189-model.pth.tar"),
        ("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth", "checkpoints/wav2lip.pth"),
        ("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/mapping_00229-model.pth.tar",
         "checkpoints/mapping_00229-model.pth.tar"),

        # New models from your script
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar",
         "checkpoints/mapping_00109-model.pth.tar"),
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors",
         "checkpoints/SadTalker_V0.0.2_256.safetensors"),
        ("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_512.safetensors",
         "checkpoints/SadTalker_V0.0.2_512.safetensors"),
    ]

    # GFPGAN weights
    gfpgan_weights = [
        ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth",
         "gfpgan/weights/alignment_WFLW_4HG.pth"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
         "gfpgan/weights/detection_Resnet50_Final.pth"),
        ("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
         "gfpgan/weights/GFPGANv1.4.pth"),
        ("https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
         "gfpgan/weights/parsing_parsenet.pth"),
    ]

    # Download essential models
    print("\n📥 Downloading Essential Models...")
    success_count = 0
    for url, filename in essential_models:
        if download_file_with_progress(url, filename):
            success_count += 1
        time.sleep(1)  # Be nice to the server

    # Download GFPGAN weights
    print("\n🎨 Downloading GFPGAN Enhancer Weights...")
    for url, filename in gfpgan_weights:
        if download_file_with_progress(url, filename):
            success_count += 1
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"✅ Download Summary:")
    print(f"   Total files processed: {len(essential_models) + len(gfpgan_weights)}")
    print(f"   Successfully downloaded: {success_count}")
    print(f"   Checkpoints folder: {os.path.abspath('checkpoints')}")
    print(f"   GFPGAN folder: {os.path.abspath('gfpgan/weights')}")
    print("=" * 60)

    # Check for missing critical files
    critical_files = [
        "checkpoints/auido2pose_00140-model.pth",
        "checkpoints/auido2exp_00300-model.pth",
        "checkpoints/epoch_20.pth",
    ]

    print("\n🔍 Checking critical files...")
    missing_files = [f for f in critical_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing critical files (may cause errors):")
        for f in missing_files:
            print(f"   - {f}")
        print("\n💡 Tip: Uncomment the legacy models in the script to download these files.")
    else:
        print("✅ All critical files are present!")


if __name__ == "__main__":
    main()