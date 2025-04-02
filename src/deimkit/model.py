from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

if TYPE_CHECKING:
    from .config import Config


def list_models():
    return [
        "deim_hgnetv2_n",
        "deim_hgnetv2_s",
        "deim_hgnetv2_m",
        "deim_hgnetv2_l",
        "deim_hgnetv2_x",
    ]


def configure_model(
    config: "Config",
    num_queries: Optional[int] = None,
    pretrained: Optional[bool] = None,
    freeze_at: Optional[int] = None,
) -> "Config":
    """
    Applies specific model parameter overrides to an existing Config object
    using explicit named arguments for selected parameters.

    Modifies the passed-in Config object.

    Args:
        config: The deimkit Config object to modify.
        num_queries: Number of object queries for the decoder (e.g., DFINETransformer).
        pretrained: Whether to load pretrained weights for the backbone.
        freeze_at: Which part to freeze in the model. Default is -1 (no freezing). If 0 freeze the stem block in the backbone.
    Returns:
        The modified Config object (the same instance passed in).

    Raises:
        ValueError: If essential parameter paths (like component types)
                    cannot be determined from the provided config object when needed.
    """

    updates: Dict[str, Any] = {}

    model_type: Optional[str] = None
    backbone_type: Optional[str] = None
    decoder_type: Optional[str] = None

    try:
        model_type = config.get("yaml_cfg.model")
        if model_type:
            backbone_type = config.get(f"yaml_cfg.{model_type}.backbone")
            decoder_type = config.get(f"yaml_cfg.{model_type}.decoder")
            if not backbone_type:
                logger.warning(
                    f"Could not determine backbone type for model '{model_type}' "
                    f"from config. 'use_pretrained_backbone' setting might not be applied."
                )
            if not decoder_type:
                logger.warning(
                    f"Could not determine decoder type for model '{model_type}' "
                    f"from config. 'num_queries' setting might not be applied."
                )
        else:
            logger.warning(
                "Could not determine 'yaml_cfg.model' from provided config. Nested settings might not be applied."
            )

    except Exception as e:
        logger.warning(
            f"Could not fully determine component types from config. "
            f"Settings might fail. Error: {e}"
        )

    def update_setting(
        setting_type: str, value: Any, type_name: Optional[str], setting_path: str
    ) -> None:
        """Helper function to update settings with consistent logging"""
        if type_name:
            key = f"yaml_cfg.{type_name}.{setting_path}"
            updates[key] = value
            logger.info(f"Setting '{key}' to: {value}")
        else:
            logger.warning(
                f"Cannot set '{setting_path}' because {setting_type} type is unknown."
            )

    if num_queries is not None:
        if decoder_type:
            update_setting("decoder", num_queries, decoder_type, "num_queries")
            # Special case for PostProcessor
            update_setting(
                "post_processor", num_queries, "PostProcessor", "num_top_queries"
            )
        else:
            logger.warning("Cannot set 'num_queries' because decoder type is unknown.")

    if pretrained is not None:
        update_setting("backbone", pretrained, backbone_type, "pretrained")

    if freeze_at is not None:
        if backbone_type:
            update_setting("backbone", freeze_at, backbone_type, "freeze_at")
            if freeze_at > 0:
                update_setting("backbone", False, backbone_type, "freeze_stem_only")
        else:
            logger.warning("Cannot set 'freeze_at' because backbone type is unknown.")

    if updates:
        config.update(**updates)  # Modifies the original config object

    return config  # Return the same object
