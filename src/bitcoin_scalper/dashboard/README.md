# Bitcoin Scalper Trading Dashboard 📊

**Complete Reconstruction - MVC Architecture with Real-Time Meta-Labeling Control**

A professional, real-time trading dashboard built with PyQt6 for monitoring and controlling the Bitcoin scalper trading bot.

> **✨ Recent Rebuild**: This dashboard has been completely reconstructed following strict MVC principles with a focus on the critical meta-labeling threshold control. All components have been verified and tested for production use.

## Features ✨

### 🎯 **Mission Control Interface**
- **Dark Cyberpunk Theme**: Optimized for macOS with elegant dark aesthetics
- **Real-time Candlestick Chart**: Live price visualization with buy/sell markers
- **Meta-Labeling Brain**: Visual confidence gauge and threshold control
- **Live Log Console**: Real-time streaming of trading decisions
- **Performance Metrics**: Balance, P&L, trade count, and win rate

### 🧠 **Meta-Labeling Control**
- **Confidence Visualization**: Progress bar showing current meta-model confidence
- **Dynamic Threshold Adjustment**: Slider and spinbox for live threshold tuning
- **Signal Status Indicator**: Visual feedback on filtered vs. passed signals
- **Real-time Feedback**: See why trades are accepted or rejected

### ⚡ **Multi-Threaded Architecture**
- **Non-Blocking UI**: Trading engine runs in separate thread
- **Smooth Performance**: UI remains responsive during intensive operations
- **Safe Shutdown**: Graceful thread termination on exit

## Installation 🚀

### Prerequisites

```bash
# Install PyQt6 and pyqtgraph
pip install PyQt6 pyqtgraph

# Or install all project dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Launch with default settings
python src/bitcoin_scalper/run_dashboard.py

# Launch with custom config and model
python src/bitcoin_scalper/run_dashboard.py \
    --config config/engine_config.yaml \
    --model models/meta_model_production.pkl

# Launch in demo mode
python src/bitcoin_scalper/run_dashboard.py --demo
```

## Usage Guide 📖

### Main Interface Components

#### 1. **Left Panel - Metrics Dashboard**
- **Balance**: Current account balance
- **P&L**: Session profit/loss (green for profit, red for loss)
- **Trades**: Total number of executed trades
- **Win Rate**: Percentage of winning trades
- **Start/Stop Buttons**: Control the trading engine

#### 2. **Center Panel - Charts & Logs**
- **Price Chart**: Real-time candlestick visualization
  - Green candles: Close > Open (bullish)
  - Red candles: Close < Open (bearish)
  - Green arrows ▲: Buy signals
  - Red arrows ▼: Sell signals
- **Log Console**: Timestamped log messages
  - 📊 Signal events
  - ✅ Trade executions
  - ⚠️ Warnings and errors
  - ℹ️ Status updates

#### 3. **Right Panel - Meta-Labeling Brain**
- **Confidence Bar**: Current meta-model confidence (0-100%)
- **Signal Indicator**: 
  - Green "BUY"/"SELL": Signal passed threshold
  - Gray "FILTERED": Signal below threshold
  - Gray "HOLD": No trading opportunity
- **Threshold Control**: Adjust meta-labeling threshold (0.0 - 1.0)
  - Higher threshold = More selective (fewer trades, higher quality)
  - Lower threshold = More aggressive (more trades, lower quality)

### Keyboard Shortcuts

- **Cmd/Ctrl + Q**: Quit application
- **Cmd/Ctrl + W**: Close window
- **Space**: Start/Stop trading (when button is focused)

## Configuration ⚙️

The dashboard uses the standard engine configuration file (`config/engine_config.yaml`):

```yaml
trading:
  symbol: "BTCUSD"
  timeframe: "M1"
  mode: "ml"
  meta_threshold: 0.6  # Meta-labeling confidence threshold

risk:
  max_drawdown: 0.05
  max_daily_loss: 0.05
  risk_per_trade: 0.01
  initial_balance: 10000.0

# ... other settings
```

### Key Configuration Parameters

- **meta_threshold**: Minimum confidence required for trade execution (0.0 - 1.0)
  - Default: 0.6 (60%)
  - Can be adjusted live via dashboard slider
- **symbol**: Trading pair (e.g., "BTCUSD", "ETHUSD")
- **timeframe**: Chart timeframe ("M1", "M5", "H1", etc.)
- **mode**: Trading mode ("ml" for machine learning, "rl" for reinforcement learning)

## Architecture 🏗️

### MVC Pattern Implementation

The dashboard follows a clean MVC (Model-View-Controller) architecture with 5 core files:

```
dashboard/
├── __init__.py              # Package initialization
├── styles.py                # VIEW - Dark theme, color constants, QSS stylesheet
│   ├── BACKGROUND_DARK (#121212)   # Main dark background constant
│   ├── TEXT_WHITE (#e0e0e0)        # Primary text color constant
│   ├── ACCENT_GREEN (#00ff00)      # Buy/profit accent constant
│   ├── ACCENT_RED (#ff0044)        # Sell/loss accent constant
│   ├── COLORS dict                  # Complete color palette
│   └── DARK_THEME_QSS              # Exported CSS stylesheet
├── worker.py                # CONTROLLER - TradingWorker thread
│   ├── TradingWorker(QThread)      # Runs engine in separate thread
│   ├── run() method                 # Infinite loop calling process_tick()
│   ├── load_ml_model()              # Model initialization
│   └── update_meta_threshold()      # Live threshold updates
├── widgets.py               # VIEW - Custom UI components
│   ├── ChartWidget (CandlestickChart)  # pyqtgraph OHLC visualization
│   ├── ControlPanel (MetaConfidencePanel)  # Meta threshold slider
│   ├── LogConsole                      # Real-time log display
│   └── StatCard                        # Metric cards
└── main_window.py           # VIEW - Main application assembly
    ├── MainWindow(QMainWindow)     # Assembles all widgets
    ├── Start/Stop buttons           # Control worker loop
    └── Signal connections           # Wires everything together

run_dashboard.py             # ENTRY POINT - Launch script
```

### Key Architecture Principles

1. **Separation of Concerns**
   - `styles.py`: Pure presentation (colors, CSS)
   - `worker.py`: Pure logic (engine orchestration)
   - `widgets.py`: Reusable UI components
   - `main_window.py`: Component assembly and wiring
   - `run_dashboard.py`: Configuration and initialization

2. **Thread Safety**
   - Trading engine runs in `TradingWorker(QThread)`
   - UI updates via Qt signals/slots (thread-safe)
   - No direct cross-thread access

3. **Real-time Updates**
   - `process_tick()` called in infinite loop
   - Results emitted as signals
   - UI updates automatically via slot connections

4. **Critical Slider Logic**
   - Meta threshold slider in `ControlPanel`
   - Range: 0.00 to 1.00 (0-100% confidence)
   - **Updates `worker.engine.meta_threshold` in real-time**
   - Connected via: `slider.valueChanged → worker.update_meta_threshold()`

### Signal Flow

```
TradingEngine (Worker Thread)
    ↓
pyqtSignal emissions
    ↓
Main UI Thread
    ↓
Update Widgets (Chart, Logs, Metrics)
```

## Troubleshooting 🔧

### Dashboard won't start

```bash
# Check PyQt6 installation
python -c "from PyQt6.QtWidgets import QApplication; print('✓ PyQt6 OK')"

# Check pyqtgraph installation
python -c "import pyqtgraph; print('✓ pyqtgraph OK')"
```

### Model not loading

- Verify model path is correct
- Check model file exists: `ls -lh models/meta_model_production.pkl`
- Ensure model is compatible with current engine version

### Chart not updating

- Check that worker thread is running (status should show "Running")
- Verify connector is fetching data (check logs for data fetch errors)
- Ensure chart widget is properly connected to worker signals

### High CPU usage

- Reduce tick frequency in worker loop (increase `time.sleep()` duration)
- Limit chart history (`max_points` in CandlestickChart)
- Disable debug logging

## Development 🛠️

### Adding Custom Indicators

Edit `widgets.py` → `CandlestickChart`:

```python
def add_indicator(self, name: str, data: List[float], color: str):
    """Add a custom indicator line to the chart."""
    pen = pg.mkPen(color=color, width=2)
    plot = self.plot_widget.plot(data, pen=pen, name=name)
    self.indicators.append(plot)
```

### Customizing Theme

Edit `styles.py` → `COLORS` dictionary:

```python
COLORS = {
    'bg_primary': '#your_color_here',
    'accent_green': '#00ff00',
    # ... customize as needed
}
```

### Adding New Metrics

1. Emit new metric from `worker.py`:
```python
self.metric_update.emit('new_metric', value)
```

2. Handle in `main_window.py`:
```python
if metric_name == 'new_metric':
    self.new_card.update_value(f"{value}")
```

## Performance Optimization 💨

- **Chart rendering**: Limited to 200 candles by default
- **Log history**: Limited to 1000 lines (auto-trimmed)
- **Update frequency**: 1 second between ticks (configurable)
- **Thread-safe operations**: All UI updates via signals/slots

## Security Notes 🔒

- **API Keys**: Never hardcode API keys in the dashboard
- **Configuration**: Store sensitive data in `config/` (gitignored)
- **Logging**: Be careful not to log sensitive information
- **Network**: Dashboard uses local connector only (no external connections)

## Contributing 🤝

When adding new features:
1. Keep UI responsive (use signals for long operations)
2. Follow the dark theme color palette
3. Add error handling for all operations
4. Update this README with new features

## License 📄

MIT License - See project root LICENSE file

## Verification ✅

All dashboard components have been tested and verified:

```bash
# Run component validation tests
python test_dashboard_components.py
```

Expected output:
```
✅ All Dashboard Components Validated Successfully!

Key Features Confirmed:
  ✓ MVC architecture with 5 files
  ✓ Color constants: BACKGROUND_DARK, TEXT_WHITE, ACCENT_GREEN, ACCENT_RED
  ✓ DARK_THEME_QSS stylesheet exported
  ✓ TradingWorker with process_tick() loop
  ✓ Meta threshold slider (0.00-1.00) updates engine in real-time
  ✓ START/STOP buttons control trading loop
  ✓ ChartWidget (pyqtgraph) for candlesticks
  ✓ LogConsole for real-time logs
  ✓ ControlPanel with meta_threshold slider
```

## Credits 👏

Built with:
- **PyQt6**: Cross-platform GUI framework
- **pyqtgraph**: High-performance plotting library
- **Bitcoin Scalper Engine**: Core trading logic

---

**Happy Trading! 🚀📈**
