"""
Example test client for the video annotation service.
This script demonstrates how to use the API.
"""
import requests
import json
import sys

def test_api(video_path, api_url="http://localhost:8000/annotate"):
    """
    Test the video annotation API with a sample video.
    
    Args:
        video_path: Path to the video file to test
        api_url: URL of the API endpoint
    """
    try:
        print(f"Testing video: {video_path}")
        print(f"API endpoint: {api_url}")
        
        with open(video_path, "rb") as video_file:
            files = {"video": (video_path, video_file, "video/mp4")}
            response = requests.post(api_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! Results:")
            print(json.dumps(result, indent=2))
            
            # Print summary
            print(f"\nSummary:")
            print(f"  Total frames: {result['total_frames']}")
            print(f"  Video: {result['video_filename']}")
            
            # Count eye states
            open_count = sum(1 for frame in result['labels_per_frame'].values() 
                           if frame['eye_state'] == 'Open')
            closed_count = sum(1 for frame in result['labels_per_frame'].values() 
                             if frame['eye_state'] == 'Closed')
            
            # Count postures
            straight_count = sum(1 for frame in result['labels_per_frame'].values() 
                                if frame['posture'] == 'Straight')
            hunched_count = sum(1 for frame in result['labels_per_frame'].values() 
                              if frame['posture'] == 'Hunched')
            
            print(f"\n Eye States:")
            print(f"  Open: {open_count} ({open_count/result['total_frames']*100:.1f}%)")
            print(f"  Closed: {closed_count} ({closed_count/result['total_frames']*100:.1f}%)")
            
            print(f"\n🪑 Posture:")
            print(f"  Straight: {straight_count} ({straight_count/result['total_frames']*100:.1f}%)")
            print(f"  Hunched: {hunched_count} ({hunched_count/result['total_frames']*100:.1f}%)")
            
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except FileNotFoundError:
        print(f"Error: Video file not found: {video_path}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running.")
        print("  Start the server with: python app.py")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <video_path>")
        print("Example: python test_client.py test_video.mp4")
    else:
        video_path = sys.argv[1]
        test_api(video_path)

