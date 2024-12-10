from typing import Annotated
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fasthtml import FastHTML
from fasthtml.common import (
    Html,
    Script,
    Head,
    Title,
    Body,
    Div,
    Form,
    Input,
    Img,
    P,
    to_xml,
)
from shad4fast import (
    ShadHead,
    Card,
    CardHeader,
    CardTitle,
    CardDescription,
    CardContent,
    CardFooter,
    Alert,
    AlertTitle,
    AlertDescription,
    Button,
    Badge,
    Separator,
    Lucide,
    Progress,
)
import base64

# Create main FastAPI app
app = FastAPI(
    title="Food Image Classification API",
    description="FastAPI application serving an ONNX model for Food image classification",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
INPUT_SIZE = (160, 160)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
LABELS = ['Acadian_Flycatcher', 'American_Crow', 'American_Goldfinch', 'American_Pipit', 'American_Redstart', 'American_Three_toed_Woodpecker', 'Anna_Hummingbird', 'Artic_Tern', 'Baird_Sparrow', 'Baltimore_Oriole', 'Bank_Swallow', 'Barn_Swallow', 'Bay_breasted_Warbler', 'Belted_Kingfisher', 'Bewick_Wren', 'Black_Tern', 'Black_and_white_Warbler', 'Black_billed_Cuckoo', 'Black_capped_Vireo', 'Black_footed_Albatross', 'Black_throated_Blue_Warbler', 'Black_throated_Sparrow', 'Blue_Grosbeak', 'Blue_Jay', 'Blue_headed_Vireo', 'Blue_winged_Warbler', 'Boat_tailed_Grackle', 'Bobolink', 'Bohemian_Waxwing', 'Brandt_Cormorant', 'Brewer_Blackbird', 'Brewer_Sparrow', 'Bronzed_Cowbird', 'Brown_Creeper', 'Brown_Pelican', 'Brown_Thrasher', 'Cactus_Wren', 'California_Gull', 'Canada_Warbler', 'Cape_Glossy_Starling', 'Cape_May_Warbler', 'Cardinal', 'Carolina_Wren', 'Caspian_Tern', 'Cedar_Waxwing', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Chipping_Sparrow', 'Chuck_will_Widow', 'Clark_Nutcracker', 'Clay_colored_Sparrow', 'Cliff_Swallow', 'Common_Raven', 'Common_Tern', 'Common_Yellowthroat', 'Crested_Auklet', 'Dark_eyed_Junco', 'Downy_Woodpecker', 'Eared_Grebe', 'Eastern_Towhee', 'Elegant_Tern', 'European_Goldfinch', 'Evening_Grosbeak', 'Field_Sparrow', 'Fish_Crow', 'Florida_Jay', 'Forsters_Tern', 'Fox_Sparrow', 'Frigatebird', 'Gadwall', 'Geococcyx', 'Glaucous_winged_Gull', 'Golden_winged_Warbler', 'Grasshopper_Sparrow', 'Gray_Catbird', 'Gray_Kingbird', 'Gray_crowned_Rosy_Finch', 'Great_Crested_Flycatcher', 'Great_Grey_Shrike', 'Green_Jay', 'Green_Kingfisher', 'Green_Violetear', 'Green_tailed_Towhee', 'Groove_billed_Ani', 'Harris_Sparrow', 'Heermann_Gull', 'Henslow_Sparrow', 'Herring_Gull', 'Hooded_Merganser', 'Hooded_Oriole', 'Hooded_Warbler', 'Horned_Grebe', 'Horned_Lark', 'Horned_Puffin', 'House_Sparrow', 'House_Wren', 'Indigo_Bunting', 'Ivory_Gull', 'Kentucky_Warbler', 'Laysan_Albatross', 'Lazuli_Bunting', 'Le_Conte_Sparrow', 'Least_Auklet', 'Least_Flycatcher', 'Least_Tern', 'Lincoln_Sparrow', 'Loggerhead_Shrike', 'Long_tailed_Jaeger', 'Louisiana_Waterthrush', 'Magnolia_Warbler', 'Mallard', 'Mangrove_Cuckoo', 'Marsh_Wren', 'Mockingbird', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Nelson_Sharp_tailed_Sparrow', 'Nighthawk', 'Northern_Flicker', 'Northern_Fulmar', 'Northern_Waterthrush', 'Olive_sided_Flycatcher', 'Orange_crowned_Warbler', 'Orchard_Oriole', 'Ovenbird', 'Pacific_Loon', 'Painted_Bunting', 'Palm_Warbler', 'Parakeet_Auklet', 'Pelagic_Cormorant', 'Philadelphia_Vireo', 'Pied_Kingfisher', 'Pied_billed_Grebe', 'Pigeon_Guillemot', 'Pileated_Woodpecker', 'Pine_Grosbeak', 'Pine_Warbler', 'Pomarine_Jaeger', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Purple_Finch', 'Red_bellied_Woodpecker', 'Red_breasted_Merganser', 'Red_cockaded_Woodpecker', 'Red_eyed_Vireo', 'Red_faced_Cormorant', 'Red_headed_Woodpecker', 'Red_legged_Kittiwake', 'Red_winged_Blackbird', 'Rhinoceros_Auklet', 'Ring_billed_Gull', 'Ringed_Kingfisher', 'Rock_Wren', 'Rose_breasted_Grosbeak', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Rusty_Blackbird', 'Sage_Thrasher', 'Savannah_Sparrow', 'Sayornis', 'Scarlet_Tanager', 'Scissor_tailed_Flycatcher', 'Scott_Oriole', 'Seaside_Sparrow', 'Shiny_Cowbird', 'Slaty_backed_Gull', 'Song_Sparrow', 'Sooty_Albatross', 'Spotted_Catbird', 'Summer_Tanager', 'Swainson_Warbler', 'Tennessee_Warbler', 'Tree_Sparrow', 'Tree_Swallow', 'Tropical_Kingbird', 'Vermilion_Flycatcher', 'Vesper_Sparrow', 'Warbling_Vireo', 'Western_Grebe', 'Western_Gull', 'Western_Meadowlark', 'Western_Wood_Pewee', 'Whip_poor_Will', 'White_Pelican', 'White_breasted_Kingfisher', 'White_breasted_Nuthatch', 'White_crowned_Sparrow', 'White_eyed_Vireo', 'White_necked_Raven', 'White_throated_Sparrow', 'Wilson_Warbler', 'Winter_Wren', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Yellow_bellied_Flycatcher', 'Yellow_billed_Cuckoo', 'Yellow_breasted_Chat', 'Yellow_headed_Blackbird', 'Yellow_throated_Vireo']

# Load the ONNX model
try:
    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("traced_models/model.onnx")
    ort_session.run(
        ["output"], {"input": np.random.randn(1, 3, *INPUT_SIZE).astype(np.float32)}
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predictions: dict  # Change to dict for class probabilities
    success: bool
    message: str

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image for model inference

    Args:
        image (PIL.Image): Input image

    Returns:
        np.ndarray: Preprocessed image array
    """
    # Convert to RGB if not already
    image = image.convert("RGB")

    # Resize
    image = image.resize(INPUT_SIZE)

    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0

    # Apply mean and std normalization
    img_array = (img_array - MEAN) / STD

    # Transpose to channel-first format (NCHW)
    img_array = img_array.transpose(2, 0, 1)

    # Add batch dimension
    img_array = np.expand_dims(img_array, 0)

    return img_array

@app.get("/", response_class=HTMLResponse)
async def ui_home():
    content = Html(
        Head(
            Title("Bird Image Classifier"),
            ShadHead(tw_cdn=True, theme_handle=True),
            Script(
                src="https://unpkg.com/htmx.org@2.0.3",
                integrity="sha384-0895/pl2MU10Hqc6jd4RvrthNlDiE9U1tWmX7WRESftEDRosgxNsQG/Ze9YMRzHq",
                crossorigin="anonymous",
            ),
        ),
        Body(
            Div(
                Card(
                    CardHeader(
                        Div(
                            CardTitle("Bird Image Classifier 🐦"),
                            Badge("AI Powered", variant="secondary", cls="w-fit"),
                            cls="flex items-center justify-between",
                        ),
                        CardDescription(
                            "Upload an image of a bird to classify it from 200+ bird species!"
                        ),
                    ),
                    CardContent(
                        Form(
                            Div(
                                Div(
                                    Input(
                                        type="file",
                                        name="file",
                                        accept="image/*",
                                        required=True,
                                        cls="mb-4 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-primary file:text-primary-foreground hover:file:bg-primary/90 file:cursor-pointer",
                                    ),
                                    P(
                                        "Drag and drop an image or click to browse",
                                        cls="text-sm text-muted-foreground text-center mt-2",
                                    ),
                                    cls="border-2 border-dashed rounded-lg p-4 hover:border-primary/50 transition-colors",
                                ),
                                Button(
                                    Lucide("sparkles", cls="mr-2 h-4 w-4"),
                                    "Classify Image",
                                    type="submit",
                                    cls="w-full",
                                ),
                                cls="space-y-4",
                            ),
                            enctype="multipart/form-data",
                            hx_post="/classify",
                            hx_target="#result",
                        ),
                        Div(id="result", cls="mt-6"),
                    ),
                    cls="w-full max-w-3xl shadow-lg",
                    standard=True,
                ),
                Div(
                    Card(
                        CardHeader(
                            CardTitle("Supported Bird Species"),
                            CardDescription("Our model can classify 200+ bird species across different families.")
                        ),
                        CardContent(
                            Div(
                                Div(
                                    Div("Flycatchers & Sparrows", cls="text-lg font-semibold mb-2"),
                                    P("Acadian Flycatcher, Baird Sparrow, Blue Grosbeak, Chipping Sparrow, Dark-eyed Junco, Grasshopper Sparrow, Henslow Sparrow, House Sparrow, Savannah Sparrow, Vesper Sparrow, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    Div("Warblers", cls="text-lg font-semibold mb-2"),
                                    P("American Redstart, Bay-breasted Warbler, Black-and-white Warbler, Black-throated Blue Warbler, Cape May Warbler, Cerulean Warbler, Hooded Warbler, Magnolia Warbler, Pine Warbler, Yellow Warbler, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    Div("Hummingbirds & Kingfishers", cls="text-lg font-semibold mb-2"),
                                    P("Anna's Hummingbird, Ruby-throated Hummingbird, Rufous Hummingbird, Belted Kingfisher, Ringed Kingfisher, Pied Kingfisher, Green Kingfisher, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    Div("Orioles & Grosbeaks", cls="text-lg font-semibold mb-2"),
                                    P("Baltimore Oriole, Orchard Oriole, Scott Oriole, Rose-breasted Grosbeak, Blue Grosbeak, Painted Bunting, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    Div("Crows, Ravens, & Jays", cls="text-lg font-semibold mb-2"),
                                    P("American Crow, Fish Crow, Common Raven, White-necked Raven, Blue Jay, Green Jay, Florida Jay, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    Div("Albatrosses, Gulls, & Terns", cls="text-lg font-semibold mb-2"),
                                    P("Black-footed Albatross, Sooty Albatross, Ivory Gull, Heermann's Gull, Western Gull, Least Tern, Arctic Tern, Common Tern, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    Div("Woodpeckers & Wrens", cls="text-lg font-semibold mb-2"),
                                    P("American Three-toed Woodpecker, Downy Woodpecker, Red-bellied Woodpecker, House Wren, Carolina Wren, Cactus Wren, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    Div("Others", cls="text-lg font-semibold mb-2"),
                                    P("Frigatebird, Mallard, Mockingbird, Nighthawk, Snowy Egret, Tree Swallow, Winter Wren, Yellow-breasted Chat, and more.", cls="text-sm"),
                                    cls="mb-4"
                                ),
                                Div(
                                    P("Tip: For best results, center the bird, ensure good lighting, and capture its unique features.", cls="text-xs text-muted-foreground italic"),
                                    cls="mt-4 text-center"
                                ),
                                cls="grid grid-cols-1 md:grid-cols-2 gap-4"
                            ),
                        ),
                        cls="w-full max-w-3xl mt-6 shadow-lg",
                        standard=True,
                    ),
                ),
                cls="container flex flex-col items-center justify-center min-h-screen p-4 space-y-6",
            ),
            cls="bg-background text-foreground",
        ),
    )
    return to_xml(content)


    
@app.post("/classify", response_class=HTMLResponse)
async def ui_handle_classify(file: Annotated[bytes, File()]):
    try:
        response = await predict(file)
        image_b64 = base64.b64encode(file).decode("utf-8")

        # Sort predictions by confidence
        sorted_predictions = sorted(response.predictions.items(), key=lambda x: x[1], reverse=True)[:5]

        # Generate HTML for predictions
        prediction_html = Div(
            Div(
                Div(cls="absolute inset-y-0 left-0 bg-primary/20 opacity-50 z-0 confidence-bar"),
                Div(
                    f"{pred[0].replace('_', ' ').title()}",
                    cls="relative z-10 text-sm font-medium"
                ),
                Div(
                    f"{pred[1]*100:.2f}%",
                    cls="text-xs text-muted-foreground"
                ),
                cls=f"prediction-row relative flex justify-between items-center p-2 border-b last:border-b-0 hover:bg-secondary/20 transition-colors",
                data_confidence=str(pred[1])
            ) for pred in sorted_predictions
        )

        # Include the original image
        image_preview = Div(
            Img(
                src=f"data:image/jpeg;base64,{image_b64}",
                cls="max-w-full max-h-64 object-contain rounded-lg mx-auto mb-4"
            ),
            cls="mb-4"
        )

        return to_xml(Div(
            image_preview,
            Div(
                Div("Top 5 Predictions", cls="text-lg font-semibold mb-3"),
                prediction_html,
                cls="bg-background border rounded-lg shadow-sm"
            )
        ))
    except Exception as e:
        return to_xml(Div(
            f"Error processing image: {str(e)}",
            cls="text-red-500 p-4 bg-red-50 rounded-lg"
        ))

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: Annotated[bytes, File(description="Image file to classify")]):
    try:
        image = Image.open(io.BytesIO(file))
        processed_image = preprocess_image(image)

        outputs = ort_session.run(
            ["output"], {"input": processed_image.astype(np.float32)}
        )

        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))

        predictions = {LABELS[i]: float(prob) for i, prob in enumerate(probabilities)}

        return PredictionResponse(
            predictions=predictions, success=True, message="Classification successful"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "model_loaded": True}, status_code=200
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)