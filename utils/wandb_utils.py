"""
W&B and Weave Integration Utilities

This module provides unified integration with Weights & Biases (W&B) and Weave
for experiment tracking, artifact management, and logging in DeepCode.

Key features:
- Independent initialization of W&B and Weave services
- Flexible artifact upload with zipped/unzipped options
- Comprehensive metadata tracking with stage/phase context
- Graceful handling of missing credentials or service unavailability
- Type-safe interface with comprehensive error handling
"""

import os
import shutil
import wandb
from wandb.sdk.wandb_run import Run 
import logging 
import weave 
from weave.trace.display.term import WeaveFormatter 
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
import glob


# Configure logging following DeepCode patterns
logger = logging.getLogger(__name__)  
  
# Apply Weave's formatter to your logger's handlers  
handler = logging.StreamHandler()  
handler.setFormatter(WeaveFormatter())  
logger.addHandler(handler)  
logger.setLevel(logging.WARNING)


def ensure_run(*args, **kwargs) -> Run:

    """
    Guarded W&B run initializer.
    If a run already exists, reuse it.
    Otherwise, call wandb.init() with provided args/kwargs.
    """
    if wandb.run is None:
        return wandb.init(*args, **kwargs)
    else:
        print(f"[W&B] Reusing existing run → id={wandb.run.id}")
        return wandb.run


def initialize_weave(
    project_name: Optional[str] = None
) -> Optional[str]:
    """
    Initialize Weave project independently.
    
    Args:
        project_name: Weave project name, defaults to environment variable
                     or 'deepcode-experiments'
        
    Returns:
        Weave project URL if successful, None otherwise
        
    Raises:
        ImportError: If weave is not installed
        Exception: If initialization fails
    """
    try:
        # Use environment variable or default
        weave_project = project_name or os.environ.get("WEAVE_PROJECT_NAME", "deepcode-experiments")
        # Initialize Weave
        weave_client = weave.init(project_name=weave_project)
        if weave_client is None:
            raise Exception("Weave initialization returned None")
        logger.info(f"Weave initialized with project: {weave_project}")
        
        # Return project URL if available
        return getattr(weave, 'get_url', lambda: "Weave project initialized")()
        
    except Exception as e:
        logger.warning(f"Failed to initialize Weave: {e}")
        return None


def initialize_wandb_run(
    project: Optional[str] = None,
    entity: Optional[str] = None,
    resume: str = "allow",
    run_id: Optional[str] = None,
    stage: str = "planning",
    init_timeout: int = 300
) -> Optional[str]:
    """
    Initialize W&B run with configuration.
    
    Args:
        project: W&B project name, defaults to WANDB_PROJECT_NAME env var
        entity: W&B entity name, defaults to WANDB_ENTITY env var
        resume: Resume mode for runs ("allow", "must", "never", "auto")
        run_id: Specific run ID to use, generates if None
        stage: Initial stage for the run
        init_timeout: Timeout for W&B initialization in seconds
        
    Returns:
        Run ID if successful, None otherwise
        
    Raises:
        ImportError: If wandb is not installed
        Exception: If initialization fails
    """
    try:
        # Use environment variables as defaults
        wandb_project = project or os.environ.get("WANDB_PROJECT_NAME", "deepcode-experiments")
        wandb_entity = entity or os.environ.get("WANDB_ENTITY")
        
        # Initialize W&B with timeout
        ensure_run(
            project=wandb_project,
            entity=wandb_entity,
            resume=resume,
            id=run_id,
            settings=wandb.Settings(init_timeout=init_timeout)
        )
        
        # Set initial stage
        wandb.config.update({"stage": stage}, allow_val_change=True)
        
        logger.info(f"W&B run initialized with ID: {wandb.run.id if wandb.run else 'N/A'}")
        return wandb.run.id if wandb.run else None
        
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        return None


def initialize_both_services(
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    weave_project: Optional[str] = None,
    stage: str = "planning"
) -> Dict[str, Optional[str]]:
    """
    Initialize both W&B and Weave services independently.
    
    Args:
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        weave_project: Weave project name
        stage: Initial stage for W&B run
        
    Returns:
        Dictionary with 'wandb_run_id' and 'weave_url' keys
    """
    result: Dict[str, Optional[str]] = {
        "wandb_run_id": None,
        "weave_url": None
    }
    
    # Initialize W&B
    result["wandb_run_id"] = initialize_wandb_run(
        project=wandb_project,
        entity=wandb_entity,
        stage=stage
    )
    
    # Initialize Weave
    result["weave_url"] = initialize_weave(project_name=weave_project)
    
    return result


def _prepare_artifact_metadata_and_aliases(
    output_dir: str,
    artifact_type: str,
    step: Optional[int],
    is_resumed: bool,
    stage: Optional[str],
    phase: Optional[str]
) -> tuple[str, Dict[str, Any], List[str]]:
    """
    Prepare metadata and aliases for W&B artifacts.
    
    Args:
        output_dir: Source directory for artifacts
        artifact_type: Type of artifact (dataset, model, etc.)
        step: Optional step number for tracking
        is_resumed: Whether this is a resumed run
        stage: Current processing stage
        phase: Current processing phase
        
    Returns:
        Tuple of (artifact_name, metadata_dict, aliases_list)
    """
    timestamp = datetime.now().isoformat()
    
    # Build metadata
    metadata: Dict[str, Any] = {
        "stage": stage or "unknown",
        "phase": phase or "unknown",
        "source_dir": output_dir,
        "timestamp": timestamp,
        "step": step,
        "is_resumed": is_resumed,
        "run_id": wandb.run.id if wandb.run else "unknown",
    }
    
    # Generate artifact name
    if stage and phase:
        artifact_name = f"paper2code-{stage}-{phase}"
    else:
        artifact_name = "paper2code-artifacts"
    
    # Build aliases list
    aliases: List[str] = ["latest"]
    if step is not None:
        aliases.append(f"step-{step}")
    if is_resumed:
        aliases.append("resumed")
    
    return artifact_name, metadata, aliases


def upload_zipped_artifact(
    output_dir: str,
    file_pattern: str,
    artifact_type: str = "dataset",
    step: Optional[int] = None,
    is_resumed: bool = False,
    stage: Optional[str] = None,
    phase: Optional[str] = None
) -> bool:
    """
    Upload artifacts to W&B as a zipped archive.
    
    Args:
        output_dir: Directory containing files to upload
        file_pattern: Glob pattern for file selection
        artifact_type: W&B artifact type classification
        step: Optional step number for tracking
        is_resumed: Whether this is a resumed run
        stage: Current processing stage
        phase: Current processing phase
        
    Returns:
        True if upload successful, False otherwise
    """
    if not wandb.run:
        logger.warning("W&B run not initialized. Skipping artifact upload.")
        return False
    
    # Find matching files
    artifact_files = glob.glob(os.path.join(output_dir, f"*{file_pattern}"))
    if not artifact_files:
        logger.warning(f"No files found matching '{file_pattern}' in {output_dir}")
        return False
    
    try:
        # Create temporary directory
        temp_dir = os.path.join(output_dir, "temp_artifacts")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Copy files to temp directory
        for file_path in artifact_files:
            shutil.copy(file_path, os.path.join(temp_dir, os.path.basename(file_path)))
        
        # Create zip archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"artifacts_step_{step}_{timestamp}.zip" if step else f"artifacts_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_name)
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", temp_dir)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        # Prepare artifact metadata
        artifact_name, metadata, aliases = _prepare_artifact_metadata_and_aliases(
            output_dir, artifact_type, step, is_resumed, stage, phase
        )
        
        # Create and log artifact
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata)
        artifact.add_file(zip_path, name=zip_name)
        
        logger.info(f"Logging zipped artifact '{artifact_name}' to W&B")
        wandb.log_artifact(artifact, aliases=aliases)
        
        # Log phase if provided
        if phase:
            wandb.log({"phase": phase})
        
        # Clean up zip file
        os.remove(zip_path)
        
        logger.info(f"Artifact '{artifact_name}' uploaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed zipped artifact upload: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_unzipped_artifact(
    output_dir: str,
    file_pattern: str,
    artifact_type: str = "dataset",
    step: Optional[int] = None,
    is_resumed: bool = False,
    stage: Optional[str] = None,
    phase: Optional[str] = None
) -> bool:
    """
    Upload artifacts to W&B without zipping.
    
    Args:
        output_dir: Directory containing files to upload
        file_pattern: Glob pattern for file selection
        artifact_type: W&B artifact type classification
        step: Optional step number for tracking
        is_resumed: Whether this is a resumed run
        stage: Current processing stage
        phase: Current processing phase
        
    Returns:
        True if upload successful, False otherwise
    """
    if not wandb.run:
        logger.warning("W&B run not initialized. Skipping artifact upload.")
        return False
    
    # Find matching files
    artifact_files = glob.glob(os.path.join(output_dir, f"*{file_pattern}"))
    if not artifact_files:
        logger.warning(f"No files found matching '{file_pattern}' in {output_dir}")
        return False
    
    try:
        # Prepare artifact metadata
        artifact_name, metadata, aliases = _prepare_artifact_metadata_and_aliases(
            output_dir, artifact_type, step, is_resumed, stage, phase
        )
        
        # Create artifact and add files
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata)
        for file_path in artifact_files:
            artifact.add_file(file_path, name=os.path.basename(file_path))
        
        # Log artifact
        logger.info(f"Logging unzipped artifact '{artifact_name}' to W&B")
        wandb.log_artifact(artifact, aliases=aliases)
        
        # Log phase if provided
        if phase:
            wandb.log({"phase": phase})
        
        logger.info(f"Artifact '{artifact_name}' uploaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed unzipped artifact upload: {e}")
        import traceback
        traceback.print_exc()
        return False


def upload_artifact(
    output_dir: str,
    file_pattern: str,
    artifact_type: str = "dataset",
    step: Optional[int] = None,
    is_resumed: bool = False,
    stage: Optional[str] = None,
    phase: Optional[str] = None,
    use_zip: bool = False
) -> bool:
    """
    Upload artifacts to W&B with configurable zipping.
    
    Args:
        output_dir: Directory containing files to upload
        file_pattern: Glob pattern for file selection
        artifact_type: W&B artifact type classification
        step: Optional step number for tracking
        is_resumed: Whether this is a resumed run
        stage: Current processing stage
        phase: Current processing phase
        use_zip: Whether to zip files before upload
        
    Returns:
        True if upload successful, False otherwise
    """
    if use_zip:
        return upload_zipped_artifact(
            output_dir=output_dir,
            file_pattern=file_pattern,
            artifact_type=artifact_type,
            step=step,
            is_resumed=is_resumed,
            stage=stage,
            phase=phase,
        )
    else:
        return upload_unzipped_artifact(
            output_dir=output_dir,
            file_pattern=file_pattern,
            artifact_type=artifact_type,
            step=step,
            is_resumed=is_resumed,
            stage=stage,
            phase=phase,
        )


def log_stage(stage: str) -> bool:
    """
    Update W&B config with current stage.
    
    Args:
        stage: Stage identifier to log
        
    Returns:
        True if successful, False otherwise
    """
    try:
        wandb.config.update({"stage": stage}, allow_val_change=True)
        logger.info(f"Updated W&B config with stage: {stage}")
        return True
    except Exception as e:
        logger.error(f"Failed to log stage: {e}")
        return False


def log_phase(phase: str) -> bool:
    """
    Log completed phase to W&B metrics.
    
    Args:
        phase: Phase identifier to log
        
    Returns:
        True if successful, False otherwise
    """
    try:
        wandb.log({"phase": phase})
        logger.info(f"Logged phase: {phase}")
        return True
    except Exception as e:
        logger.error(f"Failed to log phase: {e}")
        return False


def generic_wandb_logger(
    output_dir: str,
    stage: str,
    phase: Optional[str] = None,
    step: Optional[int] = None,
    is_resumed: bool = False,
    artifact_type: str = "dataset",
    file_pattern: str = "*.json"
) -> bool:
    """
    Generic W&B logger for artifact upload and phase tracking.
    
    Args:
        output_dir: Directory containing JSON files to upload
        stage: Current processing stage identifier
        phase: Optional current phase identifier
        step: Optional step number for tracking
        is_resumed: Whether this run resumed from a prior state
        artifact_type: W&B artifact type classification
        file_pattern: Glob pattern for file selection
        
    Returns:
        True if operations successful, False otherwise
    """
    success = True
    
    # Upload artifacts
    if not upload_artifact(
        output_dir=output_dir,
        file_pattern=file_pattern,
        artifact_type=artifact_type,
        step=step,
        is_resumed=is_resumed,
        stage=stage,
        phase=phase,
        use_zip=False
    ):
        success = False
    
    # Log phase if provided
    if phase and not log_phase(phase):
        success = False
    
    return success
