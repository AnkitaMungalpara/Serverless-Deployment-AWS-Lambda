import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from gradio.flagging import SimpleCSVLogger

class BirdClassifier:
    def __init__(self, model_path="model.pt"):
        self.device = torch.device('cpu')
        
        # Load the traced model
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the same transforms used during training/testing
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Class labels
        self.labels = ['Acadian_Flycatcher', 'American_Crow', 'American_Goldfinch', 'American_Pipit', 'American_Redstart', 'American_Three_toed_Woodpecker', 'Anna_Hummingbird', 'Artic_Tern', 'Baird_Sparrow', 'Baltimore_Oriole', 'Bank_Swallow', 'Barn_Swallow', 'Bay_breasted_Warbler', 'Belted_Kingfisher', 'Bewick_Wren', 'Black_Tern', 'Black_and_white_Warbler', 'Black_billed_Cuckoo', 'Black_capped_Vireo', 'Black_footed_Albatross', 'Black_throated_Blue_Warbler', 'Black_throated_Sparrow', 'Blue_Grosbeak', 'Blue_Jay', 'Blue_headed_Vireo', 'Blue_winged_Warbler', 'Boat_tailed_Grackle', 'Bobolink', 'Bohemian_Waxwing', 'Brandt_Cormorant', 'Brewer_Blackbird', 'Brewer_Sparrow', 'Bronzed_Cowbird', 'Brown_Creeper', 'Brown_Pelican', 'Brown_Thrasher', 'Cactus_Wren', 'California_Gull', 'Canada_Warbler', 'Cape_Glossy_Starling', 'Cape_May_Warbler', 'Cardinal', 'Carolina_Wren', 'Caspian_Tern', 'Cedar_Waxwing', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Chipping_Sparrow', 'Chuck_will_Widow', 'Clark_Nutcracker', 'Clay_colored_Sparrow', 'Cliff_Swallow', 'Common_Raven', 'Common_Tern', 'Common_Yellowthroat', 'Crested_Auklet', 'Dark_eyed_Junco', 'Downy_Woodpecker', 'Eared_Grebe', 'Eastern_Towhee', 'Elegant_Tern', 'European_Goldfinch', 'Evening_Grosbeak', 'Field_Sparrow', 'Fish_Crow', 'Florida_Jay', 'Forsters_Tern', 'Fox_Sparrow', 'Frigatebird', 'Gadwall', 'Geococcyx', 'Glaucous_winged_Gull', 'Golden_winged_Warbler', 'Grasshopper_Sparrow', 'Gray_Catbird', 'Gray_Kingbird', 'Gray_crowned_Rosy_Finch', 'Great_Crested_Flycatcher', 'Great_Grey_Shrike', 'Green_Jay', 'Green_Kingfisher', 'Green_Violetear', 'Green_tailed_Towhee', 'Groove_billed_Ani', 'Harris_Sparrow', 'Heermann_Gull', 'Henslow_Sparrow', 'Herring_Gull', 'Hooded_Merganser', 'Hooded_Oriole', 'Hooded_Warbler', 'Horned_Grebe', 'Horned_Lark', 'Horned_Puffin', 'House_Sparrow', 'House_Wren', 'Indigo_Bunting', 'Ivory_Gull', 'Kentucky_Warbler', 'Laysan_Albatross', 'Lazuli_Bunting', 'Le_Conte_Sparrow', 'Least_Auklet', 'Least_Flycatcher', 'Least_Tern', 'Lincoln_Sparrow', 'Loggerhead_Shrike', 'Long_tailed_Jaeger', 'Louisiana_Waterthrush', 'Magnolia_Warbler', 'Mallard', 'Mangrove_Cuckoo', 'Marsh_Wren', 'Mockingbird', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Nelson_Sharp_tailed_Sparrow', 'Nighthawk', 'Northern_Flicker', 'Northern_Fulmar', 'Northern_Waterthrush', 'Olive_sided_Flycatcher', 'Orange_crowned_Warbler', 'Orchard_Oriole', 'Ovenbird', 'Pacific_Loon', 'Painted_Bunting', 'Palm_Warbler', 'Parakeet_Auklet', 'Pelagic_Cormorant', 'Philadelphia_Vireo', 'Pied_Kingfisher', 'Pied_billed_Grebe', 'Pigeon_Guillemot', 'Pileated_Woodpecker', 'Pine_Grosbeak', 'Pine_Warbler', 'Pomarine_Jaeger', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Purple_Finch', 'Red_bellied_Woodpecker', 'Red_breasted_Merganser', 'Red_cockaded_Woodpecker', 'Red_eyed_Vireo', 'Red_faced_Cormorant', 'Red_headed_Woodpecker', 'Red_legged_Kittiwake', 'Red_winged_Blackbird', 'Rhinoceros_Auklet', 'Ring_billed_Gull', 'Ringed_Kingfisher', 'Rock_Wren', 'Rose_breasted_Grosbeak', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Rusty_Blackbird', 'Sage_Thrasher', 'Savannah_Sparrow', 'Sayornis', 'Scarlet_Tanager', 'Scissor_tailed_Flycatcher', 'Scott_Oriole', 'Seaside_Sparrow', 'Shiny_Cowbird', 'Slaty_backed_Gull', 'Song_Sparrow', 'Sooty_Albatross', 'Spotted_Catbird', 'Summer_Tanager', 'Swainson_Warbler', 'Tennessee_Warbler', 'Tree_Sparrow', 'Tree_Swallow', 'Tropical_Kingbird', 'Vermilion_Flycatcher', 'Vesper_Sparrow', 'Warbling_Vireo', 'Western_Grebe', 'Western_Gull', 'Western_Meadowlark', 'Western_Wood_Pewee', 'Whip_poor_Will', 'White_Pelican', 'White_breasted_Kingfisher', 'White_breasted_Nuthatch', 'White_crowned_Sparrow', 'White_eyed_Vireo', 'White_necked_Raven', 'White_throated_Sparrow', 'Wilson_Warbler', 'Winter_Wren', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Yellow_bellied_Flycatcher', 'Yellow_billed_Cuckoo', 'Yellow_breasted_Chat', 'Yellow_headed_Blackbird', 'Yellow_throated_Vireo']

    @torch.no_grad()
    def predict(self, image):
        if image is None:
            return None
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess image
        img_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Create prediction dictionary
        return {
            self.labels[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }

# Create classifier instance
classifier = BirdClassifier()

# Format available classes into HTML table - 10 per row
formatted_classes = ['<tr>']
for i, label in enumerate(classifier.labels):
    if i > 0 and i % 10 == 0:
        formatted_classes.append('</tr><tr>')
    formatted_classes.append(f'<td>{label}</td>')
formatted_classes.append('</tr>')

# Create HTML table with styling
table_html = f"""
<style>
    .food-classes-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
    }}
    .food-classes-table td {{
        padding: 6px;
        text-align: center;
        border: 1px solid var(--border-color-primary);
        font-size: 14px;
        color: var(--body-text-color);
    }}
    .food-classes-table tr td {{
        background-color: var(--background-fill-primary);
    }}
</style>
<table class="food-classes-table">
    {''.join(formatted_classes)}
</table>
"""


# Create Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    title="Bird Classification",
    description="Upload an image to classify a bird",
    flagging_mode="never",
    flagging_callback=SimpleCSVLogger(),
    article=f"Available Bird Classes:\n{table_html}"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080) 
