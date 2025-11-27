PROMPTS_CONFIG = {
    # JUST IMAGE
    "just_image": [
        {"role": "<|User|>", "content": "<image>"},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # GENERAL STATEMENT
    "generic": [
        {"role": "<|User|>", "content": "<image>\nThe industrial product shown in this image is defect-free. All components are present in the correct quantity and arrangement, satisfying all structural and logical quality constraints."},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # GENERIC WITH GROUNDING
    "grounding": [
        {"role": "<|User|>", "content": ("<image>\n"
                "<|grounding|>This image shows <|ref|>a compliant industrial product where "
                "all required components are present, correctly aligned, and free from "
                "foreign objects or placement errors<|/ref|>."
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

}