# Real Gas Fanno Flow Solver

## Overview
This project implements a numerical solver for Fanno flow in a constant-area duct, extended to include real gas effects using CoolProp.

Instead of assuming ideal gas behavior, the solver evaluates thermodynamic properties at each step, making the results more realistic for high-pressure or cryogenic applications.

---

## Features
- Marching solution along the duct length  
- Real gas property evaluation using CoolProp  
- Mach number evolution  
- Static and total pressure variation  
- Temperature variation  
- Works with different fluids (methane, nitrogen, etc.)

---

Some key characteristics:
- Total temperature remains constant  
- Total pressure decreases due to friction  
- Flow moves toward Mach 1 (choking)

---

## Dependencies

Install the required packages:

```bash
pip install numpy scipy matplotlib CoolProp 
