import os
import argparse
from datasets import load_dataset

# Full list of class names based on the label mapping
CLASS_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles"
]

def main():
    parser = argparse.ArgumentParser(description="Download a specific number of images per class from Food-101 without fully downloading the entire dataset.")
    parser.add_argument("--samples_per_class", type=int, default=10, help="Number of images to download per class. Pass -1 to download all images.")
    parser.add_argument("--output_dir", type=str, default="../data/food101", help="Output directory to save images.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face user access token to bypass rate limits and enable faster downloads.")
    
    args = parser.parse_args()

    print(f"Setting up folders in '{args.output_dir}'...")
    # Create output directories for each class
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(args.output_dir, class_name), exist_ok=True)
        
    print(f"Loading ethz/food101 (train and validation combined sequentially) in streaming mode...")
    
    # Keep track of how many images we've downloaded for each class
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    total_classes_completed = 0
    num_classes = len(CLASS_NAMES)
    
    download_all = (args.samples_per_class == -1)
    
    if download_all:
        print("Downloading ALL images for all classes...")
    else:
        print(f"Downloading {args.samples_per_class} images per class...")
    
    # Iterate sequentially through the splits
    images_processed = 0
    for split_name in ["train", "validation"]:
        dataset = load_dataset("ethz/food101", split=split_name, streaming=True, token=args.hf_token)
        
        for item in dataset:
            images_processed += 1
            
            if images_processed % 1000 == 0:
                print(f"Processed {images_processed} total items from stream...")

            label_id = item['label']
            
            # Check if we still need more samples for this class or if downloading all
            if download_all or class_counts[label_id] < args.samples_per_class:
                class_name = CLASS_NAMES[label_id]
                image = item['image']
                
                # Save the image locally as JPG
                image_path = os.path.join(args.output_dir, class_name, f"{class_counts[label_id]:04d}.jpg")
                image.save(image_path)
                
                class_counts[label_id] += 1
                
                # If not downloading all, track completion
                if not download_all:
                    if class_counts[label_id] == args.samples_per_class:
                        total_classes_completed += 1
                        print(f"Completed class: {class_name} ({total_classes_completed}/{num_classes} classes done)")
                        
                    # Once all 101 classes have 'samples_per_class' images, stop the stream
                    if total_classes_completed == num_classes:
                        print(f"Success! Downloaded {args.samples_per_class} samples for each of the 101 classes.")
                        break
        
        if not download_all and total_classes_completed == num_classes:
            break
                    
    if download_all:
        print("Success! Stream exhausted and all images have been downloaded.")

if __name__ == "__main__":
    main()
