import os
import random

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models

from torchvision.models import ResNet50_Weights
from PIL import Image


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best_waste_resnet50_weighted.pth"
DATASET_PATH = "Data2/original"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main application background */
    .stApp {
        background: linear-gradient(
            135deg,
            #0b1720 0%,
            #102a3a 55%,
            #174e59 100%
        );
        color: #f4f7fa;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #101e29;
        border-right: 1px solid #29404e;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f4f7fa;
    }

    /* Main hero section */
    .hero {
        background: linear-gradient(
            120deg,
            rgba(25, 55, 75, 0.98),
            rgba(39, 103, 111, 0.95)
        );

        border: 1px solid rgba(130, 210, 215, 0.25);
        border-radius: 24px;

        padding: 45px;
        margin-bottom: 30px;

        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.18);
    }

    .hero-tag {
        display: inline-block;

        background: rgba(100, 220, 190, 0.12);
        border: 1px solid rgba(100, 220, 190, 0.35);

        color: #8ce5cf;

        padding: 8px 14px;
        border-radius: 30px;

        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;

        margin-bottom: 18px;
    }

    .hero h1 {
        font-size: 48px;
        line-height: 1.15;
        font-weight: 800;
        color: #ffffff;

        margin-bottom: 18px;
    }

    .hero p {
        color: #d1e0e8;
        font-size: 17px;
        line-height: 1.7;

        max-width: 650px;
    }

    .hero-highlight {
        color: #8ce5cf;
    }

    /* Section headings */
    h1, h2, h3 {
        color: #f4f7fa;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: rgba(24, 43, 57, 0.85);
        border: 1px dashed #5c9da5;
        border-radius: 18px;
        padding: 18px;
    }

    /* Uploaded image */
    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(23, 43, 57, 0.8);
        border: 1px solid #365664;
        border-radius: 14px;
        padding: 18px;
    }

    /* Sidebar category cards */
    .sidebar-heading {
        color: #91d9d3;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 5px;
    }

    .sidebar-description {
        color: #9fb2bf;
        font-size: 12px;
        margin-bottom: 20px;
    }

    /* Sidebar image styling */
    [data-testid="stSidebar"] img {
        border-radius: 8px;
        object-fit: cover;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(
            90deg,
            #2c8d9b,
            #3bb6a0
        );

        color: white;
        border: none;
        border-radius: 10px;

        font-weight: 700;
    }

    .stButton button:hover {
        border: 1px solid #8ce5cf;
        color: white;
    }

    /* Divider */
    hr {
        border-color: rgba(140, 190, 200, 0.2);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_to_idx = checkpoint["class_to_idx"]

    idx_to_class = {
        index: category
        for category, index in class_to_idx.items()
    }

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_to_idx)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    # ImageNet preprocessing used by ResNet50
    eval_transform = ResNet50_Weights.DEFAULT.transforms()

    return model, idx_to_class, eval_transform


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(
    image,
    model,
    idx_to_class,
    eval_transform
):

    image = image.convert("RGB")

    image_tensor = eval_transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

    probabilities = probabilities.squeeze(0).cpu().tolist()

    predictions = []

    for index, probability in enumerate(probabilities):

        predictions.append(
            {
                "Category": idx_to_class[index],
                "Probability": probability
            }
        )

    predictions.sort(
        key=lambda item: item["Probability"],
        reverse=True
    )

    return predictions


# =========================================================
# FIND DATASET EXAMPLE IMAGE
# =========================================================

@st.cache_data
def get_category_images():

    category_images = {}

    if not os.path.isdir(DATASET_PATH):
        return category_images

    supported_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    )

    for category in os.listdir(DATASET_PATH):

        category_path = os.path.join(
            DATASET_PATH,
            category
        )

        if not os.path.isdir(category_path):
            continue

        images = [
            os.path.join(category_path, filename)
            for filename in os.listdir(category_path)
            if filename.lower().endswith(supported_extensions)
        ]

        if images:
            category_images[category] = images

    return category_images


def get_example_image(category, category_images):

    images = category_images.get(category, [])

    if not images:
        return None

    return random.choice(images)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

try:

    model, idx_to_class, eval_transform = load_model()

except Exception as error:

    st.error(
        f"Could not load the trained model: {error}"
    )

    st.stop()


category_images = get_category_images()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
    <div class="sidebar-heading">
        WASTE CLASSIFICATION
    </div>

    <h2 style="margin-top: 0;">
        ♻️ Categories
    </h2>

    <div class="sidebar-description">
        Dataset examples and model probabilities
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# HERO SECTION
# ==========================================

st.markdown(
    """
<div class="hero-content">
<div class="hero-tag">
♻️ AI POWERED WASTE SORTING
</div>

<h1>
Give waste a
<span class="hero-highlight">second chance.</span>
</h1>

<p>
Upload an image and let our trained ResNet50 model
identify its waste category.
</p>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("### Classify your waste")

st.caption(
    "Upload a JPG, JPEG, PNG, or WEBP image to begin."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    uploaded_image = Image.open(
        uploaded_file
    ).convert("RGB")

    predictions = predict_image(
        uploaded_image,
        model,
        idx_to_class,
        eval_transform
    )

    top_prediction = predictions[0]

    predicted_category = top_prediction["Category"]

    confidence = top_prediction["Probability"]


    # =====================================================
    # UPLOADED IMAGE AND PREDICTION
    # =====================================================

    st.divider()

    st.markdown("### Image analysis")

    image_column, prediction_column = st.columns(
        [1.1, 1]
    )

    with image_column:

        st.image(
            uploaded_image,
            caption="Your uploaded image",
            use_container_width=True
        )

    with prediction_column:

        st.markdown("#### Predicted category")

        st.success(
            f"🏆 {predicted_category.upper()}"
        )

        st.metric(
            label="Model confidence",
            value=f"{confidence * 100:.2f}%"
        )

        st.progress(
            float(confidence)
        )

        st.caption(
            "Confidence is the model's predicted probability, "
            "not a guarantee that the classification is correct."
        )


    # =====================================================
    # SIDEBAR CATEGORY EXAMPLES AND PROBABILITIES
    # =====================================================

    st.sidebar.divider()

    st.sidebar.markdown(
        "### Model predictions"
    )

    st.sidebar.caption(
        "Dataset examples for each predicted category."
    )

    for prediction in predictions:

        category = prediction["Category"]

        probability = prediction["Probability"]

        example_path = get_example_image(
            category,
            category_images
        )

        # Compact image + category layout
        image_column, text_column = st.sidebar.columns(
            [1, 2.5]
        )

        with image_column:

            if example_path is not None:

                example_image = Image.open(
                    example_path
                ).convert("RGB")

                st.image(
                    example_image,
                    use_container_width=True
                )

            else:

                st.write("♻️")

        with text_column:

            if category == predicted_category:

                st.markdown(
                    f"**🏆 {category.upper()}**"
                )

            else:

                st.markdown(
                    f"**{category.capitalize()}**"
                )

            st.caption(
                f"{probability * 100:.2f}% probability"
            )

            st.progress(
                float(probability)
            )

        st.sidebar.markdown(
            "<div style='height: 5px;'></div>",
            unsafe_allow_html=True
        )


else:

    st.info(
        "Upload an image above to see the model's prediction."
    )

    # Show example categories in sidebar before upload
    st.sidebar.divider()

    st.sidebar.markdown(
        "### Dataset categories"
    )

    st.sidebar.caption(
        "Upload an image to see the probabilities "
        "for each category."
    )

    for category in sorted(category_images.keys()):

        example_path = get_example_image(
            category,
            category_images
        )

        image_column, text_column = st.sidebar.columns(
            [1, 2.5]
        )

        with image_column:

            if example_path is not None:

                example_image = Image.open(
                    example_path
                ).convert("RGB")

                st.image(
                    example_image,
                    use_container_width=True
                )

        with text_column:

            st.markdown(
                f"**{category.capitalize()}**"
            )

            st.caption(
                "Dataset example"
            )