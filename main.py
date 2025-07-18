import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import threading
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
from simulation_backend import display_event


# Import your simulation backend modules
try:
    from simulation_backend import (
        SimulationConfig, NPCGenerator, DisinformationSimulator,
        create_sample_events
    )
except ImportError as e:
    print(f"Warning: Could not import simulation backend: {e}")
    # Mock classes for demonstration
    class SimulationConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class NPCGenerator:
        def __init__(self, config): pass
        def generate_npcs(self, **kwargs): return []
    
    class DisinformationSimulator:
        def __init__(self, npc_data, config): pass
        def simulate_tick(self, **kwargs): return {"day": 0, "display_event(tick)":0, "daily_summary": {"avg_trust": 0.5, "total_converted": 0, "total_new_conversions": 0, "total_interactions": 0}}
    
    def create_sample_events(): return {}


@dataclass
class SimulationState:
    """Holds the current state of the simulation"""
    is_running: bool = False
    is_paused: bool = False
    current_tick: int = 0
    total_ticks: int = 0
    logs: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.logs is None:
            self.logs = []


class ParameterValidator:
    """Validates simulation parameters"""
    
    @staticmethod
    def validate_parameters(params: Dict[str, str]) -> Dict[str, Any]:
        """Validate and convert parameters to appropriate types"""
        validated = {}
        
        try:
            validated['total_npcs'] = int(params['Total NPCs'])
            if validated['total_npcs'] <= 0:
                raise ValueError("Total NPCs must be positive")
            
            validated['total_ticks'] = int(params['Total Ticks'])
            if validated['total_ticks'] <= 0:
                raise ValueError("Total Ticks must be positive")
            
            validated['groups'] = int(params['Groups'])
            if validated['groups'] <= 0:
                raise ValueError("Groups must be positive")
            
            validated['conversion_threshold'] = float(params['Conversion Threshold'])
            if not 0 <= validated['conversion_threshold'] <= 1:
                raise ValueError("Conversion Threshold must be between 0 and 1")
            
            return validated
            
        except ValueError as e:
            raise ValueError(f"Invalid parameter: {e}")


class SimulationLogger:
    """Handles logging functionality"""
    
    def __init__(self, text_widget: tk.Text):
        self.text_widget = text_widget
        self.log_entries = []
    
    def log(self, message: str, level: str = "INFO"):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {level}: {message}"
        
        self.text_widget.insert(tk.END, formatted_msg + "\n")
        self.text_widget.see(tk.END)
        self.log_entries.append({"timestamp": timestamp, "level": level, "message": message})
    
    def clear(self):
        """Clear all log entries"""
        self.text_widget.delete(1.0, tk.END)
        self.log_entries.clear()
    
    def export_logs(self, filename: str):
        """Export logs to file"""
        with open(filename, "w") as f:
            for entry in self.log_entries:
                f.write(f"{entry['timestamp']} - {entry['level']}: {entry['message']}\n")


class PlotManager:
    """Manages plotting functionality"""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.figure = None
        self.canvas = None
        self.plot_frame = None
    
    def create_embedded_plot(self, logs: List[Dict[str, Any]]):
        """Create an embedded plot in the GUI"""
        if not logs:
            messagebox.showerror("Error", "No data to plot")
            return
        
        # Create or update plot frame
        if self.plot_frame:
            self.plot_frame.destroy()
        
        self.plot_frame = ttk.LabelFrame(self.parent_frame, text="Simulation Results")
        self.plot_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create figure
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # Extract data
        days = [log["day"] for log in logs]
        avg_trust = [log["daily_summary"]["avg_trust"] for log in logs]
        conversions = [log["daily_summary"]["total_converted"] for log in logs]
        interactions = [log["daily_summary"]["total_interactions"] for log in logs]
        
        # Plot data
        ax1.plot(days, avg_trust, label="Average Trust", color='blue', marker='o')
        ax1.set_ylabel("Average Trust")
        ax1.set_title("Trust Levels Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(days, conversions, label="Converted NPCs", color='red', marker='s')
        ax2.plot(days, interactions, label="Interactions", color='green', marker='^')
        ax2.set_xlabel("Days")
        ax2.set_ylabel("Count")
        ax2.set_title("Activity Metrics")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        self.figure.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_external_plot(self, logs: List[Dict[str, Any]]):
        """Show plot in external window"""
        if not logs:
            messagebox.showerror("Error", "No data to plot")
            return
        
        days = [log["day"] for log in logs]
        avg_trust = [log["daily_summary"]["avg_trust"] for log in logs]
        conversions = [log["daily_summary"]["total_converted"] for log in logs]
        interactions = [log["daily_summary"]["total_interactions"] for log in logs]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(days, avg_trust, 'b-o', label="Average Trust")
        plt.xlabel("Days")
        plt.ylabel("Trust Level")
        plt.title("Average Trust Over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(days, conversions, 'r-s', label="Converted NPCs")
        plt.xlabel("Days")
        plt.ylabel("Count")
        plt.title("Converted NPCs Over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(days, interactions, 'g-^', label="Interactions")
        plt.xlabel("Days")
        plt.ylabel("Count")
        plt.title("Daily Interactions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        new_conversions = [log["daily_summary"]["total_new_conversions"] for log in logs]
        plt.bar(days, new_conversions, color='orange', alpha=0.7)
        plt.xlabel("Days")
        plt.ylabel("New Conversions")
        plt.title("New Conversions per Day")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class SimulationApp:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Disinformation Simulation GUI v2.0")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.state = SimulationState()
        self.simulator = None
        self.validator = ParameterValidator()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
        # Create widgets
        self.create_widgets()
        
        # Initialize logger after widgets are created
        self.logger = SimulationLogger(self.log_text)
        self.plot_manager = PlotManager(self.root)
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Parameters frame
        param_frame = ttk.LabelFrame(self.root, text="Simulation Parameters")
        param_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.entries = {}
        fields = [
            ("Total NPCs", "1000", "Number of NPCs in the simulation"),
            ("Total Ticks", "10", "Number of simulation days"),
            ("Groups", "5", "Number of social groups"),
            ("Conversion Threshold", "0.2", "Threshold for belief conversion (0-1)")
        ]
        
        for i, (label, default, tooltip) in enumerate(fields):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            
            entry = ttk.Entry(param_frame, width=15)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries[label] = entry
            
            # Add tooltip
            ttk.Label(param_frame, text=tooltip, font=("Arial", 8), foreground="gray").grid(
                row=i, column=2, sticky="w", padx=5, pady=2
            )
        
        # Control buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Pause", command=self.pause_simulation, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_simulation, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="Reset", command=self.reset).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Show Plot", command=self.show_external_plot).grid(row=0, column=4, padx=5)
        ttk.Button(button_frame, text="Embed Plot", command=self.show_embedded_plot).grid(row=0, column=5, padx=5)
        ttk.Button(button_frame, text="Export Data", command=self.export_data).grid(row=0, column=6, padx=5)
        ttk.Button(button_frame, text="Export Logs", command=self.export_logs).grid(row=0, column=7, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, columnspan=8, sticky="ew", pady=5)
        
        # Log frame
        log_frame = ttk.LabelFrame(self.root, text="Simulation Logs")
        log_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(text_frame, height=15, width=100, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, sticky="ew", padx=5, pady=2)
    
    def update_status(self, message: str):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def update_progress(self, current: int, total: int):
        """Update progress bar"""
        progress = (current / total) * 100 if total > 0 else 0
        self.progress_var.set(progress)
        self.root.update_idletasks()
    
    def reset(self):
        """Reset the simulation"""
        self.state = SimulationState()
        if hasattr(self, 'logger'):
            self.logger.clear()
        self.update_status("Ready")
        self.progress_var.set(0)
        
        # Reset button states
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
    
    def start_simulation(self):
        """Start the simulation in a separate thread"""
        try:
            # Validate parameters
            param_values = {label: entry.get() for label, entry in self.entries.items()}
            validated_params = self.validator.validate_parameters(param_values)
            
            # Update state
            self.state.is_running = True
            self.state.is_paused = False
            self.state.current_tick = 0
            self.state.total_ticks = validated_params['total_ticks']
            self.state.logs = []
            
            # Update UI
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            self.update_status("Starting simulation...")
            
            # Start simulation thread
            thread = threading.Thread(target=self.run_simulation, args=(validated_params,))
            thread.daemon = True
            thread.start()
            
        except ValueError as e:
            messagebox.showerror("Parameter Error", str(e))
        except Exception as e:
            self.logger.log(f"Error starting simulation: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to start simulation: {e}")
    
    def pause_simulation(self):
        """Pause/resume the simulation"""
        if self.state.is_running:
            self.state.is_paused = not self.state.is_paused
            if self.state.is_paused:
                self.pause_button.configure(text="Resume")
                self.update_status("Simulation paused")
            else:
                self.pause_button.configure(text="Pause")
                self.update_status("Simulation resumed")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.state.is_running = False
        self.state.is_paused = False
        self.update_status("Stopping simulation...")
        
        # Reset button states
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled", text="Pause")
        self.stop_button.configure(state="disabled")
    
    def run_simulation(self, params: Dict[str, Any]):
        """Run the simulation (called in separate thread)"""
        try:
            # Create configuration
            config = SimulationConfig(
                total_npcs=params['total_npcs'],
                total_ticks=params['total_ticks'],
                conversion_threshold=params['conversion_threshold'],
                num_groups=params['groups'],
                ingroup_communication_bias=0.7,
                random_seed=42
            )
            
            self.logger.log("Generating NPCs...")
            self.update_status("Generating NPCs...")
            
            # Generate NPCs
            generator = NPCGenerator(config)
            npc_data = generator.generate_npcs(
                npc_count=config.total_npcs,
                age_based_ratio=0.6,
                trust_range=(0.3, 0.9),
                num_groups=config.num_groups
            )
            
            # Create simulator
            events = create_sample_events()
            self.simulator = DisinformationSimulator(npc_data, config)
            
            self.logger.log("Starting simulation...")
            self.update_status("Running simulation...")
            
            # Run simulation
            for tick in range(config.total_ticks):
                # Check if simulation should stop
                if not self.state.is_running:
                    break
                
                # Wait while paused
                while self.state.is_paused and self.state.is_running:
                    threading.Event().wait(0.1)
                
                if not self.state.is_running:
                    break
                
                # Update progress
                self.update_progress(tick, config.total_ticks)
                
                # Run simulation tick
                tick_log = self.simulator.simulate_tick(
                    tick,
                    events.get(tick, []),
                    trust_erosion_multiplier=0.05,
                    group_trust_modifier=1.5,
                    ingroup_bias=0.8
                )
                
                self.state.logs.append(tick_log)
                self.state.current_tick = tick
                
                # Log results
                summary = tick_log["daily_summary"]
                log_line = (f"Day {tick + 1}: {display_event(tick)} \n "
                            f"                       Avg Trust: {summary['avg_trust']:.4f}, "
                            f"Converted: {summary['total_converted']}, "
                            f"New Conversions: {summary['total_new_conversions']}, "
                            f"Interactions: {summary['total_interactions']}")
                self.logger.log(log_line)
                
                # Small delay to prevent UI freezing
                threading.Event().wait(0.1)
            
            # Simulation completed
            if self.state.is_running:
                self.logger.log("Simulation completed successfully!", "SUCCESS")
                self.update_status("Simulation completed")
                self.update_progress(config.total_ticks, config.total_ticks)
            else:
                self.logger.log("Simulation stopped by user", "INFO")
                self.update_status("Simulation stopped")
            
        except Exception as e:
            self.logger.log(f"Error during simulation: {e}", "ERROR")
            self.logger.log(traceback.format_exc(), "ERROR")
            self.update_status("Simulation failed")
        finally:
            # Reset button states
            self.root.after(0, self.reset_buttons)
    
    def reset_buttons(self):
        """Reset button states (called from main thread)"""
        self.start_button.configure(state="normal")
        self.pause_button.configure(state="disabled", text="Pause")
        self.stop_button.configure(state="disabled")
    
    def show_external_plot(self):
        """Show plot in external window"""
        self.plot_manager.show_external_plot(self.state.logs)
    
    def show_embedded_plot(self):
        """Show plot embedded in GUI"""
        self.plot_manager.create_embedded_plot(self.state.logs)
    
    def export_data(self):
        """Export simulation data to JSON"""
        if not self.state.logs:
            messagebox.showwarning("Warning", "No data to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                export_data = {
                    "simulation_info": {
                        "total_ticks": self.state.total_ticks,
                        "completed_ticks": len(self.state.logs),
                        "export_time": datetime.now().isoformat()
                    },
                    "logs": self.state.logs
                }
                
                with open(filename, "w") as f:
                    json.dump(export_data, f, indent=2)
                
                self.logger.log(f"Data exported to: {filename}", "SUCCESS")
                messagebox.showinfo("Success", f"Data exported to {filename}")
                
            except Exception as e:
                self.logger.log(f"Error exporting data: {e}", "ERROR")
                messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def export_logs(self):
        """Export log messages to text file"""
        if not hasattr(self, 'logger') or not self.logger.log_entries:
            messagebox.showwarning("Warning", "No logs to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                self.logger.export_logs(filename)
                messagebox.showinfo("Success", f"Logs exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export logs: {e}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.state.is_running:
            if messagebox.askokcancel("Quit", "Simulation is running. Do you want to quit?"):
                self.stop_simulation()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
