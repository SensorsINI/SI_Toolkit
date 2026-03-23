"""
Network parsing utilities for hls4ml conversion.

This module provides functions to parse network names and extract configuration
parameters needed for VHDL package generation.
"""

import re


def parse_network_name(net_name):
    """
    Parse network name to extract input and output neuron counts.
    
    Args:
        net_name (str): Network name in format like 'Dense-7IN-32H1-32H2-1OUT-0'
        
    Returns:
        tuple: (input_neurons, output_neurons)
        
    Example:
        >>> parse_network_name('Dense-7IN-32H1-32H2-1OUT-0')
        (7, 1)
    """
    # Pattern: Dense-7IN-32H1-32H2-1OUT-0
    # Extract input neurons (7IN) and output neurons (1OUT)
    input_match = re.search(r'(\d+)IN', net_name)
    output_match = re.search(r'(\d+)OUT', net_name)
    
    input_neurons = int(input_match.group(1)) if input_match else 7
    output_neurons = int(output_match.group(1)) if output_match else 1
    
    return input_neurons, output_neurons


def extract_data_bits_from_precision(precision_str):
    """
    Extract data bit width from ap_fixed precision string.
    
    Args:
        precision_str (str): Precision string like 'ap_fixed<14,2>'
        
    Returns:
        int: Total number of bits
        
    Example:
        >>> extract_data_bits_from_precision('ap_fixed<14,2>')
        14
    """
    # Pattern: 'ap_fixed<14,2>' -> total bits = 14
    match = re.search(r'ap_fixed<(\d+),', precision_str)
    return int(match.group(1)) if match else 12


def parse_network_config(net_name, precision_config):
    """
    Parse network configuration to extract all needed parameters.
    
    Args:
        net_name (str): Network name
        precision_config (dict): Precision configuration from config_hls.yml
        
    Returns:
        dict: Parsed network parameters
    """
    input_neurons, output_neurons = parse_network_name(net_name)
    
    # Extract data bit widths from precision settings
    input_data_bits = extract_data_bits_from_precision(precision_config['input_and_output'])
    output_data_bits = extract_data_bits_from_precision(precision_config['input_and_output'])
    
    return {
        'input_neurons': input_neurons,
        'output_neurons': output_neurons,
        'input_data_bits': input_data_bits,
        'output_data_bits': output_data_bits
    }
