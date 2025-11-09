"""Business logic workflows for hard-negative mining."""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import curator, engine
from .models import (
    ExportConfig,
    ExportResult,
    InferenceConfig,
    InferenceResult,
    LinkMethod,
    ModelSource,
)

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """Orchestrates hard-negative mining workflows."""
    
    def __init__(self, target_species: str = "RADR"):
        """
        Initialize miner with target species.
        
        Args:
            target_species: Species label to search for in predictions
        """
        self.target_species = target_species
        logger.info(f"Initialized HardNegativeMiner for species: {target_species}")
    
    def run_inference(self, config: InferenceConfig) -> InferenceResult:
        """
        Run analyzer inference and return results.
        
        Args:
            config: Inference configuration
        
        Returns:
            InferenceResult with dataframe and metadata
        """
        logger.info(f"Starting inference: {config.model_source.value}, input={config.input_dir}")
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.error(f"Configuration validation failed: {errors}")
            return InferenceResult(success=False, errors=errors)
        
        try:
            stamp = int(time.time())
            tmp_model_file = None
            out_root = None
            
            # Execute based on model source
            if config.model_source == ModelSource.EXPERIMENT_CANONICAL:
                logger.info(f"Running experiment canonical: {config.selected_experiment}")
                proc, out_root, cmd = engine.run_inference_for_experiment_stream(
                    config.selected_experiment,
                    config.input_dir
                )
            
            elif config.model_source == ModelSource.EXPERIMENT_MODEL_FILE:
                logger.info(f"Running with model file: {config.model_path}")
                out_root = config.input_dir.parent / 'low_quality_inference' / f'ui_{stamp}'
                out_root.mkdir(parents=True, exist_ok=True)
                proc, out_root, cmd = engine.run_analyzer_cli_stream(
                    input_dir=config.input_dir,
                    out_dir=out_root,
                    model_path=config.model_path
                )
            
            else:  # UPLOADED_FILE
                logger.info("Running with uploaded model")
                # Save uploaded file to temp location
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='_' + config.uploaded_model.name)
                tmp.write(config.uploaded_model.getvalue())
                tmp.flush()
                tmp.close()
                tmp_model_file = Path(tmp.name)
                
                out_root = config.input_dir.parent / 'low_quality_inference' / f'ui_{stamp}'
                out_root.mkdir(parents=True, exist_ok=True)
                proc, out_root, cmd = engine.run_analyzer_cli_stream(
                    input_dir=config.input_dir,
                    out_dir=out_root,
                    model_path=tmp_model_file
                )
            
            # Return process handle for streaming (caller will handle logs)
            return InferenceResult(
                success=True,
                output_path=out_root,
                command=cmd,
                dataframe=None  # Will be populated after process completes
            )
        
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return InferenceResult(success=False, errors=[str(e)])
        
        finally:
            # Cleanup temp model file if created
            if tmp_model_file:
                try:
                    tmp_model_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp model file: {e}")
    
    def collect_results(
        self,
        output_dir: Path,
        input_dir: Path,
        save_path: Optional[Path] = None
    ) -> InferenceResult:
        """
        Collect and aggregate analyzer results.
        
        Args:
            output_dir: Directory containing analyzer CSV outputs
            input_dir: Input directory for file matching
            save_path: Optional path to save aggregated CSV
        
        Returns:
            InferenceResult with aggregated dataframe
        """
        logger.info(f"Collecting results from: {output_dir}")
        
        try:
            # Aggregate per-file max confidence
            df = engine.collect_per_file_max(output_dir, target_label=self.target_species)
            logger.info(f"Collected {len(df)} predictions")
            
            # Match to input files
            df_matched = curator.match_files(df, input_dir)
            logger.info(f"Matched {len(df_matched)} files")
            
            if df_matched.empty:
                return InferenceResult(
                    success=False,
                    errors=["No matches found between analyzer output and input files"]
                )
            
            # Save if path provided
            csv_path = None
            if save_path:
                df_matched.to_csv(save_path, index=False)
                csv_path = save_path
                logger.info(f"Saved results to: {csv_path}")
            
            return InferenceResult(
                success=True,
                dataframe=df_matched,
                output_path=output_dir,
                csv_path=csv_path
            )
        
        except Exception as e:
            logger.error(f"Failed to collect results: {e}", exc_info=True)
            return InferenceResult(success=False, errors=[str(e)])
    
    def load_from_csv(self, csv_path: Path, input_dir: Path) -> InferenceResult:
        """
        Load results from existing CSV file.
        
        Args:
            csv_path: Path to CSV file with per-file confidences
            input_dir: Directory to match files against
        
        Returns:
            InferenceResult with loaded dataframe
        """
        logger.info(f"Loading CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Parse columns
            conf_col = next(
                (c for c in df.columns if 'confidence' in c.lower() or 'conf' in c.lower()),
                None
            )
            file_col = next(
                (c for c in df.columns if c.lower() == 'file' or c.lower().endswith('file')),
                df.columns[0] if len(df.columns) > 0 else None
            )
            
            if not conf_col or not file_col:
                return InferenceResult(
                    success=False,
                    errors=["Could not detect confidence and file columns in CSV"]
                )
            
            # Normalize column names
            df_normalized = df[[file_col, conf_col]].copy()
            df_normalized.columns = ['File', 'radr_max_confidence']
            
            # Match files
            df_matched = curator.match_files(df_normalized, input_dir)
            
            if df_matched.empty:
                return InferenceResult(
                    success=False,
                    errors=["No matches found between CSV rows and files in input folder"]
                )
            
            logger.info(f"Loaded and matched {len(df_matched)} files")
            
            return InferenceResult(
                success=True,
                dataframe=df_matched
            )
        
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}", exc_info=True)
            return InferenceResult(success=False, errors=[f"Failed to parse CSV: {e}"])
    
    def export_files(
        self,
        files: List[Path],
        config: ExportConfig
    ) -> ExportResult:
        """
        Export selected files to destination.
        
        Args:
            files: List of file paths to export
            config: Export configuration
        
        Returns:
            ExportResult with success status and counts
        """
        logger.info(f"Exporting {len(files)} files using {config.method.value}")
        
        if not files:
            return ExportResult(
                success=False,
                errors=["No files selected to export"]
            )
        
        try:
            dest = config.get_destination()
            dest.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            errors = []
            
            for file_path in files:
                target = dest / file_path.name
                
                if target.exists():
                    logger.debug(f"Skipping existing file: {target}")
                    continue
                
                try:
                    if config.link_method == LinkMethod.COPY:
                        import shutil
                        shutil.copy2(file_path, target)
                    elif config.link_method == LinkMethod.HARDLINK:
                        import os
                        os.link(file_path, target)
                    elif config.link_method == LinkMethod.SYMLINK:
                        target.symlink_to(file_path)
                    
                    success_count += 1
                
                except Exception as e:
                    error_msg = f"Failed to {config.link_method.value} {file_path.name}: {e}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
            
            logger.info(f"Exported {success_count}/{len(files)} files to: {dest}")
            
            return ExportResult(
                success=True,
                files_exported=success_count,
                total_files=len(files),
                destination=dest,
                errors=errors
            )
        
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return ExportResult(
                success=False,
                total_files=len(files),
                errors=[str(e)]
            )
    
    def aggregate_results(self, search_paths: List[Path]) -> Optional[pd.DataFrame]:
        """
        Aggregate results from multiple directories.
        
        Args:
            search_paths: List of paths to search for result CSVs
        
        Returns:
            Aggregated DataFrame or None if no results found
        """
        logger.info(f"Aggregating results from {len(search_paths)} paths")
        
        try:
            master = curator.aggregate_results([str(p) for p in search_paths])
            
            if master is None or master.empty:
                logger.warning("No results found to aggregate")
                return None
            
            logger.info(f"Aggregated {len(master)} unique files")
            return master
        
        except Exception as e:
            logger.error(f"Aggregation failed: {e}", exc_info=True)
            return None
