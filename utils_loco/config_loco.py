PROMPTS_CONFIG = {
    # -------------------
    # JUST IMAGE
    # -------------------
    "just_image": [
        {"role": "<|User|>", "content": "<image>"},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # GENERAL STATEMENT
    # -------------------
    "generic": [
        {"role": "<|User|>", "content": "<image>\nThe industrial product shown in this image is defect-free. All components are present in the correct quantity and arrangement, satisfying all structural and logical quality constraints."},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # GENERIC WITH GROUNDING
    # -------------------
    "grounding": [
        {"role": "<|User|>", "content": ("<image>\n"
                "<|grounding|>This image shows <|ref|>a compliant industrial product where "
                "all required components are present, correctly aligned, and free from "
                "foreign objects or placement errors<|/ref|>."
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # SPECIFIC WITH GROUNDING
    # -------------------
    # Breakfast Box
    "grounding_s_breakfast_box": [
        {
            "role": "<|User|>",
            "content": (
                "<image>\n"
                "<|grounding|>This image shows <|ref|>a compliant breakfast box divided into two specific zones. "
                "The left compartment strictly contains exactly two tangerines and one nectarine. "
                "The right compartment is filled with a precise mix of cereals, banana chips, and almonds, "
                "maintaining a fixed ratio and relative position of ingredients<|/ref|>."
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Screw Bag
    "grounding_s_screw_bag": [
        {
            "role": "<|User|>",
            "content": (
                "<image>\n"
                "<|grounding|>This image shows <|ref|>a compliant transparent screw bag containing a strict hardware set: "
                "exactly two washers, two nuts, one long screw, and one short screw. "
                "All components are present inside the bag without any foreign objects<|/ref|>."
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Pushpins
    "grounding_s_pushpins": [
        {
            "role": "<|User|>",
            "content": (
                "<image>\n"
                "<|grounding|>This image shows <|ref|>a compliant box of pushpins with a fully intact grid structure. "
                "Every single compartment is occupied by exactly one pushpin. "
                "There are no empty slots, no compartments with multiple pins, and all pins are undamaged<|/ref|>."
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Splicing Connectors
    "grounding_s_splicing_connectors": [
        {
            "role": "<|User|>",
            "content": (
                "<image>\n"
                "<|grounding|>This image shows <|ref|>a compliant splicing connector assembly exhibiting mirror symmetry. "
                "It features exactly two connectors with the same number of clamps, linked by exactly one cable. "
                "The cable color corresponds to the clamp count, and the cable terminates in the same relative position on both ends<|/ref|>."
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Juice Bottle
    "grounding_s_juice_bottle": [
        {
            "role": "<|User|>",
            "content": (
                "<image>\n"
                "<|grounding|>This image shows <|ref|>a compliant juice bottle with a standardized fill level. "
                "It carries exactly two labels: a central label with a fruit icon that strictly matches the color of the liquid, "
                "and a lower label displaying the text '100% Juice'<|/ref|>."
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # GENERIC IN ASSISTANT
    # -------------------
    "assistant_generic": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|><image>\n"
                "Inspect <|ref|>this industrial assembly<|/ref|> to verify structural integrity "
                "and logical compliance (correct component count and arrangement)."
            )
        },
        {
            "role": "<|Assistant|>",
            "content": (
                "Inspection Report: Confirmed. The assembly is defect-free. "
                "All components are present in the correct quantity and position. "
                "No foreign objects, structural damages, or logical violations detected."
            )
        }
    ],

    # -------------------
    # ASSISTANT PER CLASS - UNSTRUCTURED
    # -------------------
    # Breakfast Box
    "assistant_u_breakfast_box": [
        {"role": "<|User|>",
         "content": "<|grounding|><image>\nInspect <|ref|>this breakfast box<|/ref|> to verify the correct arrangement and quantity of the food items."},
        {"role": "<|Assistant|>",
         "content": ("Inspection passed. The box is in perfect condition. It contains exactly two tangerines and one nectarine on the left. The right compartment is filled with cereals and toppings, "
                     "strictly maintaining the correct quantity and component ratio. All items are positioned correctly and are free from defects.")}
    ],

    # Screw Bag
    "assistant_u_screw_bag": [
        {"role": "<|User|>",
         "content": "<|grounding|><image>\nInspect <|ref|>this bag of screws<|/ref|> to confirm the correct count and condition of the hardware components."},
        {"role": "<|Assistant|>",
         "content": "Inspection passed. The bag contains exactly two washers, two nuts, one long screw, and one short screw. All components are present, undamaged, and free from foreign objects."}
    ],

    # Pushpins
    "assistant_u_pushpins": [
        {"role": "<|User|>",
         "content": "<|grounding|><image>\nInspect <|ref|>this box of pushpins<|/ref|> to verify compartment logic and structural integrity."},
        {"role": "<|Assistant|>",
         "content": ("Inspection passed. The box structure and dividers are fully intact. "
                    "Each compartment is populated by exactly one pushpin. "
                    "There are no empty compartments and no overcrowded sections. "
                    "All pins are in perfect condition with undamaged tips.")}
    ],

    # Splicing Connectors
    "assistant_u_splicing_connectors": [
        {"role": "<|User|>",
         "content": "<|grounding|><image>\nInspect <|ref|>this splicing connector assembly<|/ref|> to check for symmetry and cable integrity."},
        {"role": "<|Assistant|>",
         "content": ("Inspection passed. The assembly features two symmetric connectors linked by exactly one cable. "
                    "The cable color corresponds correctly to the number of clamps. "
                    "The entire structure is intact, the cable is undamaged, and all components are correctly aligned.")}
    ],

    # Juice Bottle
    "assistant_u_juice_bottle": [
        {"role": "<|User|>",
         "content": "<|grounding|><image>\nInspect <|ref|>this juice bottle<|/ref|> to verify the label, icon, and fill level."},
        {"role": "<|Assistant|>",
         "content": ("Inspection passed. The bottle is in perfect condition. "
                "It displays the correct fruit icon label in the center and the text label at the bottom. "
                "The fruit icon consistently matches the liquid color. "
                "The fill level is precise, and the bottle is free from structural defects or contamination.")}
    ],

    # -------------------
    # ASSISTANT PER CLASS - STRUCTURED JSON
    # -------------------
    # Breakfast Box
    "assistant_s_breakfast_box": [
        {"role": "<|User|>", "content": "<|grounding|><image>\nInspect <|ref|>this breakfast box<|/ref|> and output the quality control report in JSON format, verifying counts and arrangement."},
        {"role": "<|Assistant|>",
         "content": """```json
         {
         "status": "PASS",
         "left_compartment": {
         "tangerines": 2,
         "nectarines": 1,
         "position": "correct"
         },
         "right_compartment": {
         "contents": ["cereals", "toppings"],
         "ratio": "correct",
         "distribution": "uniform"
         },
         "defects_detected": false
         }
         ```"""
         }
    ],

    # Screw Bag
    "assistant_s_screw_bag": [
        {"role": "<|User|>", "content": "<|grounding|><image>\nInspect <|ref|>this bag of screws<|/ref|> and provide a JSON inventory check."},
        {"role": "<|Assistant|>", "content": """```json
         {
         "status": "PASS",
         "inventory": {
         "washers": 2,
         "nuts": 2,
         "long_screws": 1,
         "short_screws": 1
         },
         "logic_check": {
         "missing_parts": false,
         "extra_parts": false
         },
         "structural_integrity": "intact"
         }
         ```"""
         }
    ],

    # Pushpins
    "assistant_s_pushpins": [
        {"role": "<|User|>", "content": "<|grounding|><image>\nInspect <|ref|>this box of pushpins<|/ref|> and validate the compartment logic in JSON format."},
        {"role": "<|Assistant|>",
         "content": """```json
         {
         "status": "PASS",
         "grid_layout": {
         "dividers": "intact",
         "occupancy": "exactly_one_per_slot"
         },
         "anomalies": {
         "empty_slots": 0,
         "overcrowded_slots": 0,
         "damaged_tips": 0
         }
         }
         ```"""
         }
    ],

    # Splicing Connectors
    "assistant_s_splicing_connectors": [
        {
            "role": "<|User|>",
            "content": "<|grounding|><image>\nInspect <|ref|>this splicing connector assembly<|/ref|> and output a JSON validation report checking clamps, wiring logic, and symmetry."
        },
        {
            "role": "<|Assistant|>",
            "content": """```json
{
  "status": "PASS",
  "assembly_composition": {
    "connector_count": 2,
    "cable_count": 1,
    "clamp_consistency": "equal_number_on_both_connectors"
  },
  "logical_constraints": {
    "wiring_logic": "cable_color_corresponds_to_clamp_count",
    "geometric_constraint": "same_relative_termination_position",
    "global_symmetry": "mirror_symmetric"
  },
  "condition": "defect_free"
}
```"""
        }
    ],

    # Juice Bottle
    "assistant_s_juice_bottle": [
        {
            "role": "<|User|>",
            "content": "<|grounding|><image>\nInspect <|ref|>this juice bottle<|/ref|> and output a validation report in JSON."
        },
        {
            "role": "<|Assistant|>",
            "content": """```json
{
  "status": "PASS",
  "visual_elements": {
    "center_label": "fruit_icon",
    "bottom_label": "text_label"
  },
  "logic_check": {
    "icon_matches_liquid": true,
    "fill_level": "precise"
  },
  "defects": "none"
}
```"""
        }
    ],

    # -------------------
    # SELF-KNOWLEDGE DISTILLATION
    # -------------------
    # Breakfast Box
    "distilled_breakfast_box": [
        {
            "role": "<|User|>",
            "content": "<|grounding|><image>\nYou are a Quality Inspector. Provide your official assessment of <|ref|>this breakfast box<|/ref|>."
        },
        {
            "role": "<|Assistant|>",
            "content": """Technical Report: Golden Sample Breakfast Box

Left Compartment:
The left compartment contains exactly two tangerines and one nectarine.
The tangerines have a vibrant orange color with a slightly glossy surface, indicating ripeness. They exhibit a smooth texture with minor natural imperfections typical of fresh citrus fruits.
The nectarine has a gradient of colors from yellow to red, suggesting it is ripe and ready to eat. Its skin appears smooth but with slight dimples, which is characteristic of nectarines.

Right Compartment:
The right compartment is filled with a mix of cereals, banana chips, and almonds.
The cereal mix consists of various grains, including oats and possibly other types of nuts or seeds. The colors range from light brown to golden, indicating a toasted or roasted preparation. The distribution is even, ensuring each bite will contain a balanced mix of ingredients.
The banana chips are sliced into uniform pieces, showcasing a pale yellow color with some darker spots where they were exposed to air during drying. Their edges appear slightly curled, which is typical for dried bananas.
Almonds are scattered throughout the mixture, adding a contrasting dark brown color. They are whole and unprocessed, providing a crunchy texture to complement the softer elements.

Container Material:
The container is made of white plastic, likely polystyrene, designed for single-use packaging. It has a rectangular shape with rounded corners and a clear partition separating the two compartments.
The material appears clean and free of any defects, meeting hygiene standards required for food packaging.

Lighting and Background:
The image is taken under bright, even lighting, which enhances the visibility of the colors and textures of the food items. There are no harsh shadows, ensuring all details are clearly visible.
The background is a solid black color, which contrasts sharply with the white container and colorful food items, making them stand out prominently.

Visual Standard:
The overall presentation adheres to a high standard of cleanliness and organization. Each item is neatly placed within its designated section, demonstrating attention to detail and adherence to quality control measures.
The combination of fresh fruits and nutritious cereals reflects a balanced and appealing meal option, suitable for consumers seeking a healthy breakfast choice.

This detailed analysis confirms that the golden sample breakfast box meets all specified manufacturing constraints and visual standards, serving as an exemplary model for future production."""
        }
    ],

# 2. Screw Bag 
    "distilled_screw_bag": [
        {
            "role": "<|User|>",
            "content": "<|grounding|><image>\nYou are a Quality Inspector. Verify the inventory and condition of <|ref|>this screw bag<|/ref|>."
        },
        {
            "role": "<|Assistant|>",
            "content": """Technical Report: Golden Sample Documentation

Subject: Inspection of Screw Bag Packaging

I. Contents Verification
- Washers: Two washers found inside the bag.
- Nuts: Two nuts found inside the bag.
- Screws: One long screw and one short screw found inside the bag.
- Total Components: The total count matches the packing list provided. No additional or missing components were observed.

II. Visual Attributes
- Hardware:
  - Washers: Made of metal with a shiny metallic sheen, reflecting light uniformly. Threads appear smooth and well-defined.
  - Nuts: Also made of metal with a similar shiny metallic sheen. Threads are sharp and precise.
  - Screws: Both screws exhibit a metallic sheen with clear, defined threads. The long screw has a longer length compared to the short screw.
- Plastic Bag:
  - Transparent with slight reflections indicating its glossy texture.
  - Folds are minimal, suggesting careful handling during packaging.
  - Texture appears smooth without any visible tears or abrasions.
- Background Surface:
  - The components rest on a black, non-reflective surface, providing a stark contrast to the metallic sheen of the hardware.
  - The bag itself is placed flat, ensuring stability and preventing any tilt that could obscure visibility.

III. Condition Check
- Scratches/Corrosion: No visible scratches, corrosion, or foreign particles were detected on the hardware.
- Overall Condition: All components appear new and unused, meeting the standards for a golden sample.

V. Conclusion
Based on the inspection, the golden sample screw bag meets all quality assurance criteria.
The contents match the packing list, and there are no defects or missing components.
The hardware's metallic sheen and thread definition indicate high manufacturing quality.
The transparent plastic bag's condition further supports the integrity of the packaging."""
        }
    ],

    # 3. Pushpin 
    "distilled_pushpin": [
        {
            "role": "<|User|>",
            "content": "<|grounding|><image>\nnYou are a Quality Inspector. Provide your official assessment of <|ref|>this pushpin box<|/ref|>."
        },
        {
            "role": "<|Assistant|>",
            "content": """Technical Report: 
Assembly Rules Verification:
This sample consists of 15 compartments, each containing exactly one yellow pushpin. There are no empty slots or overcrowded compartments. All compartments are uniformly filled, adhering to the specified assembly rules.
Product Appearance Description:
Pushpins: Each pushpin has a bright yellow plastic head with a smooth, rounded shape. The metallic tips are sharp and pointed, designed for secure insertion into various materials. The pushpins exhibit uniform coloration and size, indicating consistent manufacturing quality.
Storage Box: The storage box is made of transparent plastic, allowing visibility of the contents without opening it. It features a grid of dividers that separate each compartment, ensuring individual pushpins do not touch or move during transport. The lid has hinges at the top, enabling easy access while maintaining the box's structural integrity. Reflections on the plastic surface suggest a clean, high-quality finish.
Orientation of Pins Within Slots:
The pushpins are oriented randomly within their respective compartments, with no specific alignment pattern. This random orientation does not affect the functionality or structural integrity of the pushpins.
Structural Integrity Certification:
The dividers appear robust and well-constructed, capable of securely holding the pushpins in place without bending or warping. The metallic tips of the pushpins are straight and uncorrupted, indicating they have been manufactured to precise specifications.
Conclusion:
This sample meets all specified criteria for quality and assembly. The pushpins are uniformly colored and sized, the storage box is structurally sound, and the orientation of the pins within the compartments does not impact functionality. This documentation confirms this sample as an ideal representation of the pushpin assembly process."""
        }
    ],

}