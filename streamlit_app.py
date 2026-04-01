import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import glob

# Import the core system components (ensure these are in your PYTHONPATH or same dir)
from src.orchestrator import FaceRecognitionSystem

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Biometric Review Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CACHED SYSTEM INITIALIZATION
# ==========================================
@st.cache_resource
def load_system():
    """Loads the ML models and FAISS DB into memory only once."""
    # NOTE: No Streamlit UI calls (like st.toast) are allowed inside here!
    system = FaceRecognitionSystem(similarity_threshold=0.45)
    
    # We also need a place to store the actual images for visualization,
    # because FAISS only stores the 512D mathematical vectors.
    image_store = {} 
    
    return system, image_store

# Initialize system globally with UI feedback OUTSIDE the cached function
with st.spinner("Initializing ML Engine & Loading Models..."):
    fr_system, img_store = load_system()
st.toast("System Ready", icon="✅")


def populate_from_local_folder(base_folder_path):
    """Enrolls faces from a local directory structure into the FAISS DB."""
    if not os.path.isdir(base_folder_path):
        st.error(f"Directory not found: {base_folder_path}")
        return
        
    subdirs = [d for d in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, d))]
    if not subdirs:
        st.warning(f"No celebrity subfolders found in {base_folder_path}")
        return

    # Initialize progress tracking UI
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    enrolled_count = 0
    total_subdirs = len(subdirs)
    
    with st.spinner("Extracting embeddings from local dataset..."):
        for i, celeb_name in enumerate(subdirs):
            celeb_dir = os.path.join(base_folder_path, celeb_name)
            
            # Find all images in this folder (handling different extensions)
            image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                image_paths.extend(glob.glob(os.path.join(celeb_dir, ext)))
            
            for img_path in image_paths:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue
                    
                # Create a unique ID for each image
                user_id = f"USR_{celeb_name.replace(' ', '_').upper()}_{enrolled_count}"
                metadata = {"name": celeb_name, "source": "Local_Dataset"}
                
                # Enroll into Vector DB
                success = fr_system.enroll_live(user_id, img_bgr)
                if success:
                    fr_system.live_db.metadata[user_id] = metadata
                    
                    # Store a downsized version for the UI to prevent RAM overflow
                    h, w = img_bgr.shape[:2]
                    scale = 200 / max(h, w)
                    img_display = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
                    img_store[user_id] = img_display
                    
                    enrolled_count += 1
            
            # Update UI elements dynamically
            progress = (i + 1) / total_subdirs
            progress_bar.progress(progress)
            status_text.text(f"Processing Folder: {celeb_name} ({i+1}/{total_subdirs}) | Total Faces Enrolled: {enrolled_count}")
            
    # Clear progress tracking UI when finished
    progress_bar.empty()
    status_text.empty()
    st.session_state['db_populated'] = True
    st.success(f"✅ Successfully enrolled {enrolled_count} faces from {total_subdirs} folders!")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def process_uploaded_image(upload) -> np.ndarray:
    """Converts Streamlit UploadedFile to OpenCV BGR format."""
    image = Image.open(upload).convert('RGB')
    image_np = np.array(image)
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Converts OpenCV BGR back to RGB for Streamlit display."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def populate_mock_database(num_identities=5):
    """Enrolls samples from the LFW dataset into the FAISS DB."""
    if st.session_state.get('db_populated', False):
        return

    images, targets, target_names = preload_lfw_gallery()
    
    with st.spinner("Populating FAISS Database with reference identities..."):
        enrolled_count = 0
        for i in range(len(target_names)):
            if enrolled_count >= num_identities:
                break
                
            # Get the first image of this person
            person_idx = targets == i
            if not np.any(person_idx): continue
            
            img_float = images[person_idx][0]
            img_bgr = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            user_id = f"USR_{target_names[i].replace(' ', '_').upper()}"
            metadata = {"name": target_names[i], "source": "LFW_Dataset"}
            
            # Enroll into Vector DB
            success = fr_system.enroll_live(user_id, img_bgr)
            if success:
                # Add metadata to FAISS manually for demo retrieval
                fr_system.live_db.metadata[user_id] = metadata
                # Store the actual image so we can render it later
                img_store[user_id] = img_bgr
                enrolled_count += 1
                
        st.session_state['db_populated'] = True
        st.success(f"Successfully populated database with {enrolled_count} identities!")

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.title("⚙️ System Controls")
st.sidebar.markdown("Adjust parameters for the similarity search.")

top_k = st.sidebar.slider("Top-K Results", min_value=1, max_value=10, value=3)
threshold = st.sidebar.slider(
    "Similarity Threshold", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="Matches below this Cosine Similarity score will be flagged as 'Low Confidence'."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Local Dataset Management")

# Text input for the user to provide the local folder path
dataset_path = st.sidebar.text_input("Dataset Folder Path", value="./Celebrity Face Image Dataset")

if st.sidebar.button("🚀 Load Local Dataset"):
    populate_from_local_folder(dataset_path)

st.sidebar.info(f"**Current Gallery Size:** {fr_system.live_db.index.ntotal} records")

# ==========================================
# MAIN DASHBOARD UI
# ==========================================
st.title("👁️ Biometric Review Dashboard")
st.markdown("Upload a probe image to search the Vector Database for similar identities.")

# File Uploader
probe_upload = st.file_uploader("Upload Probe Face (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if probe_upload is not None:
    col1, col2 = st.columns([1, 2])
    
    # Process the probe image
    probe_bgr = process_uploaded_image(probe_upload)
    
    with col1:
        st.subheader("Probe Image")
        st.image(bgr_to_rgb(probe_bgr), use_container_width=True, caption="Target to search")
    
    with col2:
        st.subheader("Analysis & Extraction")
        with st.spinner("Extracting 512D Embedding..."):
            start_time = time.time()
            processed_probe = fr_system.preprocessor.process(probe_bgr, source="LIVE")
            extract_time = time.time() - start_time
            
            if not processed_probe:
                st.error("❌ No valid face detected or image failed quality/liveness checks.")
            else:
                st.success(f"✅ Face detected and aligned. (Took {extract_time*1000:.0f} ms)")
                
                # Show bounding box on image (optional visual flair)
                bbox = processed_probe["bbox"].astype(int)
                display_img = probe_bgr.copy()
                cv2.rectangle(display_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                st.image(bgr_to_rgb(display_img), use_container_width=True, caption="Detected Face ROI")

    # ==========================================
    # SEARCH & RESULTS SECTION
    # ==========================================
    if processed_probe:
        st.markdown("---")
        st.header(f"🔍 Top {top_k} Matches found in Vector DB")
        
        probe_embedding = processed_probe["embedding"]
        
        # Query FAISS
        start_search = time.time()
        matched_ids, scores = fr_system.live_db.search(probe_embedding, top_k=top_k)
        search_time = time.time() - start_search
        
        st.caption(f"Vector search completed in {search_time*1000:.2f} ms")
        
        if not matched_ids:
            st.warning("The gallery is empty! Please load the sample gallery from the sidebar or enroll users.")
        else:
            # Create a dynamic grid of columns for the results
            cols = st.columns(len(matched_ids))
            
            for i, (match_id, score) in enumerate(zip(matched_ids, scores)):
                with cols[i]:
                    is_verified = score >= threshold
                    
                    # Result Card UI
                    if is_verified:
                        st.success(f"**Rank {i+1}**")
                    else:
                        st.error(f"**Rank {i+1}**")
                        
                    # Fetch metadata and image
                    meta = fr_system.live_db.get_metadata(match_id) or {}
                    name = meta.get("name", "Unknown")
                    ref_image = img_store.get(match_id)
                    
                    if ref_image is not None:
                        st.image(bgr_to_rgb(ref_image), use_container_width=True)
                    else:
                        st.info("No visual reference stored.")
                        
                    st.markdown(f"**ID:** `{match_id}`")
                    st.markdown(f"**Name:** {name}")
                    
                    # Score formatting
                    score_pct = score * 100
                    st.metric(label="Cosine Similarity", value=f"{score_pct:.1f}%", 
                              delta="PASS" if is_verified else "FAIL", 
                              delta_color="normal" if is_verified else "inverse")