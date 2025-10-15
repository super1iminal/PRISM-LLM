import logging 
import tempfile
from typing import List 
import subprocess
import os
import traceback


class PrismVerifier:
    def __init__(self, prism_bin_path: str):
        self.prism_bin_path = prism_bin_path
        self.logger = logging.getLogger(__name__)
        self.temp_files = [] 

    def verify_property(self, model_str: str, property_str: str) -> List[float]:
        model_path = None
        prop_path = None
        try:
            # Save model to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.nm', delete=False) as model_file:
                model_file.write(model_str)
                model_path = model_file.name
                self.temp_files.append(model_path)

            # Save properties to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.props', delete=False) as prop_file:
                prop_file.write(property_str)
                prop_path = prop_file.name
                self.temp_files.append(prop_path)

            # Debug log the model and properties
            self.logger.debug("PRISM Model:")
            self.logger.debug(model_str)
            self.logger.debug("Properties:")
            self.logger.debug(property_str)

            # Construct PRISM command
            cmd = [
                self.prism_bin_path,
                model_path,
                prop_path,
                "-explicit",
                "-javamaxmem", "4g",
                "-maxiters", "1000000",  
                "-power", 
                "-verbose",
                "-exportstates", "logs/states.txt"  
            ]

            # Run PRISM
            self.logger.debug(f"Running PRISM command: {' '.join(cmd)}")
            result = subprocess.run(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, 
                                 text=True)

            # Log PRISM output for debugging
            self.logger.debug("PRISM stdout:")
            self.logger.debug(result.stdout)
            self.logger.debug("PRISM stderr:")
            self.logger.debug(result.stderr)

            # Parse results
            probabilities = []
            for line in result.stdout.split('\n'):
                if "Result:" in line:
                    try:
                        value_str = line.split(':')[1].strip().split()[0]
                        prob = float(value_str)
                        probabilities.append(prob)
                        self.logger.debug(f"Parsed probability: {prob}")
                    except (IndexError, ValueError) as e:
                        self.logger.error(f"Failed to parse line: {line}")
                        self.logger.error(f"Error: {str(e)}")
                        continue

            if not probabilities:
                self.logger.error("No probabilities found in PRISM output")
                return [0.0] * 8  

            return probabilities

        except Exception as e:
            self.logger.error(f"Verification error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return [0.0] * 8

        finally:
            # Cleanup temporary files
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    self.logger.error(f"Failed to delete temporary file {temp_file}: {str(e)}")
            self.temp_files = []