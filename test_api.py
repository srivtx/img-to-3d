"""Quick integration test for the 3D generation API."""

import requests
import time
import os

BASE_URL = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("Health:", r.json())
    assert r.status_code == 200


def test_generate():
    # Create a dummy image
    from PIL import Image
    img = Image.new("RGB", (512, 512), color="red")
    img.save("/tmp/test_image.png")
    
    with open("/tmp/test_image.png", "rb") as f:
        r = requests.post(
            f"{BASE_URL}/generate-3d",
            files={"image": ("test.png", f, "image/png")}
        )
    
    print("Generate response:", r.json())
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    
    # Poll until complete
    for _ in range(30):
        time.sleep(1)
        status = requests.get(f"{BASE_URL}/jobs/{job_id}").json()
        print(f"Status: {status['status']} ({status['progress_percent']}%)")
        if status["status"] in ("completed", "failed"):
            break
    
    assert status["status"] == "completed"
    assert status["preview_url"] is not None
    assert status["final_url"] is not None
    print("Success! Preview:", status["preview_url"])
    print("Final:", status["final_url"])


if __name__ == "__main__":
    test_health()
    test_generate()
