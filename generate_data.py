import pandas as pd
import numpy as np
import random
import os

# Configuration
NUM_SAMPLES = 2000
DATA_FILE = "support_tickets_data.csv"

# Categories with semantic indicators (more realistic, less direct keywords)
categories = {
    "Billing & Payments": {
        "indicators": ["money", "charge", "cost", "pay", "card", "account balance", "transaction", "statement", "due", "amount"],
        "templates": [
            "I noticed an issue with my account - the amount doesn't match what I expected.",
            "There seems to be a problem with the recent charge on my account.",
            "Can you help me understand this transaction from last week?",
            "I was charged multiple times for the same order.",
            "My subscription hasn't been renewed properly.",
            "The amount I was charged seems incorrect.",
            "I need to update my payment information.",
            "There's a discrepancy in my billing statement.",
            "I didn't authorize this charge on my account.",
            "Can you process a refund for my recent purchase?",
            "My credit card declined but I was still charged.",
            "How do I change my subscription settings?",
            "I have an issue with my recent purchase.",
            "Something is wrong with my order.",
            "Can you help me with my purchase?",
            "I need assistance with an issue."
        ]
    },
    "Technical Support": {
        "indicators": ["error", "crash", "bug", "slow", "doesn't work", "broken", "issue", "problem", "malfunction", "timeout"],
        "templates": [
            "The application keeps crashing whenever I try to use this feature.",
            "I'm experiencing performance issues with the platform.",
            "There's a bug in the system that prevents me from completing my task.",
            "The service seems to be down or experiencing technical difficulties.",
            "I'm getting error messages when trying to access certain features.",
            "The page loads extremely slowly for me.",
            "Something isn't functioning correctly in the application.",
            "I can't seem to access the platform from my device.",
            "The system returned an unexpected error code.",
            "The feature you mentioned isn't working as described.",
            "I'm having trouble connecting to the service.",
            "The application froze while I was using it.",
            "I have an issue with my account - something isn't working right.",
            "There's a problem I need help with.",
            "I need assistance with an issue.",
            "Can you help me resolve this problem?"
        ]
    },
    "Product Inquiry": {
        "indicators": ["product", "item", "model", "version", "feature", "spec", "capability", "include", "available", "option"],
        "templates": [
            "I'd like to know more about your latest product offerings.",
            "What features are included in this version?",
            "Is this item available in different sizes or colors?",
            "Can you tell me more about the specifications?",
            "Do you have information about what's included in the package?",
            "What are the main features of this product?",
            "Is there a newer version available?",
            "What options are available for customization?",
            "Can you provide more details about the product?",
            "What's the difference between these product versions?",
            "Is this item still being manufactured?",
            "What accessories are compatible with this product?",
            "I have a question about an item.",
            "Can you help me with some information?",
            "I need assistance regarding something.",
            "I'm looking for more details about something."
        ]
    },
    "Returns & Refunds": {
        "indicators": ["return", "exchange", "refund", "back", "damaged", "defective", "broken", "wrong", "incorrect", "received"],
        "templates": [
            "I received a damaged item and would like to return it.",
            "The item I received wasn't what I ordered.",
            "I'd like to return this purchase for a refund.",
            "Can I exchange this for a different option?",
            "The product arrived defective and doesn't work.",
            "How do I initiate a return?",
            "I need to send this back as it's not what I expected.",
            "The wrong item was sent to me.",
            "Can I get a refund for this purchase?",
            "I'd like to return this item and get my money back.",
            "The shipment included an incorrect item.",
            "This product doesn't meet my expectations and I'd like to return it.",
            "I have an issue with my recent order.",
            "Something is wrong with what I received.",
            "I need help with my purchase.",
            "Can you assist me with something?"
        ]
    },
    "Account Management": {
        "indicators": ["account", "login", "password", "email", "profile", "access", "verify", "register", "user", "credentials"],
        "templates": [
            "I'm having trouble logging into my account.",
            "I need to change my account password.",
            "Can you help me recover my account access?",
            "I forgot my login credentials.",
            "I'd like to update my account information.",
            "How do I verify my email address?",
            "I think my account has been compromised.",
            "Can I change the email associated with my account?",
            "I'm unable to reset my password.",
            "How do I update my profile information?",
            "I need help accessing my account.",
            "Can you help me set up two-factor authentication?",
            "I have an issue with my account.",
            "Something's wrong with my access.",
            "I need assistance with my account.",
            "Can you help me with my account issue?"
        ]
    },
    "General/Irrelevant": {
        "indicators": ["hello", "hi", "thanks", "question", "help", "inquiry", "information", "know", "tell", "want"],
        "templates": [
            "Hi there, just wanted to reach out.",
            "I'm looking for some general information.",
            "What can you tell me about your company?",
            "I have a question that might not fit your usual categories.",
            "Can you provide some information about something?",
            "I'm interested in learning more about your services.",
            "Do you have any recommendations for me?",
            "I'd like to provide feedback about something.",
            "Just checking in to see how things are going.",
            "I have a general inquiry for your team.",
            "Can you help me with some information?",
            "I wanted to ask about something in general.",
            "I need assistance with something.",
            "I have a question for you.",
            "Can you help me out?",
            "I'm reaching out about something."
        ]
    }
}

urgency_keywords = {
    "High": ["immediately", "urgent", "asap", "emergency", "critical", "now", "right now", "can't wait", "important", "soon", "quickly", "need", "have to", "must", "priority", "please resolve immediately"],
    "Medium": ["should", "would like", "prefer", "want", "issue", "problem", "help", "assistance", "trouble", "regarding"],
    "Low": ["when possible", "eventually", "at some point", "no rush", "whenever", "thanks", "question", "information", "inquiry", "general"]
}

def generate_ticket(category):
    """Generate a realistic ticket with semantic content and some ambiguity."""
    template = random.choice(categories[category]["templates"])
    
    # Add noise - sometimes add irrelevant context
    if random.random() < 0.3:
        noise = random.choice([
            " By the way, the weather is nice today.",
            " I hope you're having a good day.",
            " This is my first time contacting support.",
            " I'm a longtime customer.",
            " I appreciate your help with this.",
            " Looking forward to your response.",
            " Thanks for your assistance.",
            ""
        ])
        template = template + noise
    
    # Random capitalization for realism
    if random.random() > 0.7:
        template = template.lower()
    
    return template

def assign_urgency(text, category):
    """Assign urgency based on keywords with minimal noise for better accuracy."""
    lower_text = text.lower()
    
    # Check urgency indicators - order matters, check High first
    if any(keyword in lower_text for keyword in urgency_keywords["High"]):
        if random.random() < 0.03:  # Very low noise for High priority
            return random.choice(list(urgency_keywords.keys()))
        return "High"
    
    if any(keyword in lower_text for keyword in urgency_keywords["Low"]):
        if random.random() < 0.05:
            return random.choice(list(urgency_keywords.keys()))
        return "Low"
    
    # Default to Medium for ambiguous cases
    if random.random() < 0.05:
        return random.choice(list(urgency_keywords.keys()))
    return "Medium"

def main():
    print(f"Generating {NUM_SAMPLES} synthetic tickets...")
    
    data = []
    all_categories = list(categories.keys())
    
    for _ in range(NUM_SAMPLES):
        true_category = random.choice(all_categories)
        ticket_text = generate_ticket(true_category)
        urgency = assign_urgency(ticket_text, true_category)
        
        # Add category noise - 10% chance to assign wrong category label (reduced from 15% for better accuracy)
        assigned_category = true_category
        if random.random() < 0.10:
            assigned_category = random.choice(all_categories)
        
        data.append({
            "ticket_id": _ + 1000,
            "text": ticket_text,
            "category": assigned_category,
            "urgency": urgency
        })
    
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"Data saved to {DATA_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()
