import logging
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

# Import the orchestrator we built earlier
from src.orchestrator import FaceRecognitionSystem

logger = logging.getLogger("DatasetRunner")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def main():
    print("\n" + "="*50)
    print("1. DOWNLOADING / LOADING LFW DATASET")
    print("="*50)
    # Fetch LFW dataset. 
    # color=True gets RGB images. resize=1.0 keeps original 250x250 resolution.
    # min_faces_per_person=2 ensures we can use 1 for enrollment and 1 for testing.
    lfw_people = fetch_lfw_people(min_faces_per_person=2, color=True, resize=1.0)
    
    images = lfw_people.images # Shape: (N, 250, 250, 3) in RGB
    target = lfw_people.target
    target_names = lfw_people.target_names

    print(f"Successfully loaded {len(images)} images belonging to {len(target_names)} different people.")

    print("\n" + "="*50)
    print("2. INITIALIZING PRODUCTION FACE REC SYSTEM")
    print("="*50)
    fr_system = FaceRecognitionSystem()

    print("\n" + "="*50)
    print("3. SELECTING TEST SUBJECTS")
    print("="*50)
    
    # Let's pick two different people for our tests
    person_a_idx = 0  # e.g., Colin Powell
    person_b_idx = 1  # e.g., George W Bush
    
    person_a_name = target_names[person_a_idx]
    person_b_name = target_names[person_b_idx]
    
    # Get all images for Person A and Person B
    person_a_images = images[target == person_a_idx]
    person_b_images = images[target == person_b_idx]

    # Convert images from RGB (sklearn default) to BGR (cv2/InsightFace default)
    # Note: sklearn images are float32 in range [0, 1]. OpenCV/Insightface needs uint8 [0, 255]
    def format_img(img_float):
        img_uint8 = (img_float * 255).astype(np.uint8)
        return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

    # Person A images
    img_a_enroll = format_img(person_a_images[0])
    img_a_probe  = format_img(person_a_images[1]) # Different photo of Person A
    
    # Person B image (Imposter)
    img_b_imposter = format_img(person_b_images[0])

    print(f"Subject A: {person_a_name} (Using 2 distinct photos)")
    print(f"Subject B (Imposter): {person_b_name} (Using 1 photo)")

    def visualize_result(img_enrolled, img_probe, title, is_match):
        """Helper to plot the enrolled image vs probe image with color-coded results."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Convert BGR (OpenCV) back to RGB for Matplotlib displaying
        axes[0].imshow(cv2.cvtColor(img_enrolled, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Enrolled Image (Reference)")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_probe, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Probe Image (Input)")
        axes[1].axis('off')
        
        color = 'green' if is_match else 'red'
        match_text = "VERIFIED (MATCH)" if is_match else "FAILED (NO MATCH)"
        
        fig.suptitle(f"{title}\nSystem Result: {match_text}", color=color, fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.show()

    print("\n" + "="*50)
    print("4. ENROLLMENT PHASE (Subject A)")
    print("="*50)
    user_id = f"USR_{person_a_name.replace(' ', '_').upper()}"
    metadata = {"name": person_a_name, "dob": "1950-01-01"} # Mock DOB
    
    success_doc = fr_system.enroll_document(user_id, img_a_enroll, metadata)
    success_live = fr_system.enroll_live(user_id, img_a_enroll)
    
    if not (success_doc and success_live):
        print(f"Failed to enroll {person_a_name}. Exiting.")
        return

    print("\n" + "="*50)
    print("5. VERIFICATION SCENARIOS")
    print("="*50)

    print(f"\n--- SCENARIO 1: TRUE MATCH (Live vs Live) ---")
    print(f"Probing with a DIFFERENT photo of {person_a_name}...")
    res1 = fr_system.verify(
        input_image=img_a_probe, 
        target_user_id=user_id, 
        mode="LIVE_TO_LIVE"
    )
    visualize_result(img_a_enroll, img_a_probe, "Scenario 1: True Match (Live vs Live)", res1)

    print(f"\n--- SCENARIO 2: TRUE MATCH (Live vs Doc + Metadata) ---")
    print(f"Probing with a DIFFERENT photo of {person_a_name} + Correct Metadata...")
    res2 = fr_system.verify(
        input_image=img_a_probe, 
        target_user_id=user_id, 
        mode="LIVE_TO_DOC",
        input_metadata={"name": person_a_name, "dob": "1950-01-01"}
    )
    visualize_result(img_a_enroll, img_a_probe, "Scenario 2: True Match (Live vs Doc)", res2)

    print(f"\n--- SCENARIO 3: IMPOSTER ATTEMPT (Live vs Live) ---")
    print(f"Probing with a photo of {person_b_name} claiming to be {person_a_name}...")
    res3 = fr_system.verify(
        input_image=img_b_imposter, 
        target_user_id=user_id, 
        mode="LIVE_TO_LIVE"
    )
    visualize_result(img_a_enroll, img_b_imposter, "Scenario 3: Imposter Attempt", res3)

if __name__ == "__main__":
    main()