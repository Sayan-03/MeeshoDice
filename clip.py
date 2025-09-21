import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel

# -----------------------------
# 1. Example Dataset (Image + Text)
# -----------------------------
class ProductDataset(Dataset):
    def __init__(self, image_paths, texts, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.processor(images=self.image_paths[idx], return_tensors="pt")["pixel_values"].squeeze(0)
        text = self.processor(text=[self.texts[idx]], return_tensors="pt", padding=True, truncation=True)
        return {
            "pixel_values": image,
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0)
        }

# -----------------------------
# 2. Contrastive Loss (InfoNCE)
# -----------------------------
def clip_contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    # Normalize
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Similarity matrix
    logits = (image_embeds @ text_embeds.T) / temperature
    labels = torch.arange(len(logits), device=logits.device)

    loss_img = nn.CrossEntropyLoss()(logits, labels)
    loss_txt = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_img + loss_txt) / 2

# -----------------------------
# 3. Training Loop
# -----------------------------
def train_clip(image_paths, texts, epochs=5, batch_size=8, lr=5e-6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CLIP
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Dataset + Loader
    dataset = ProductDataset(image_paths, texts, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds

            # Loss
            loss = clip_contrastive_loss(image_embeds, text_embeds)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model, processor

# -----------------------------
# 4. Example Usage
# -----------------------------
if __name__ == "__main__":
    # Dummy example: replace with real product images + titles
    image_paths = ["C:/Users/ghsay/OneDrive/Desktop/shirt.png"]
    texts = ["Red cotton shirt for men", "Men’s red cotton shirt", "Traditional silk saree", "Silk saree for women"]

    trained_model, trained_processor = train_clip(image_paths, texts, epochs=3)
    torch.save(trained_model.state_dict(), "clip_smartmerge.pth")
