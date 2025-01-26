Braude College of Engineering, Karmiel 
Capstone Project Phase A,B

Project Name:EXAMINING THE PERFORMANCE OF SATELLITE PROPAGATORS FOR AUTONOMOUS SPACE-SITUATIONAL-AWARENESS SATELLITES 24-2-R-13

Supervisor:  Elad Denenberg

Mahran AbedEllatif - 209120815 Mahran.Abedellatif@e.braude.ac.il

Shadi AbedAlkream 209120096 Shadi.Abd.Alkream@e.braude.ac.il


Project describtion (Abstract):
The exponential growth in satellite deployments and space debris has elevated the risk of collisions, underscoring the need for precise state propagation algorithms to ensure safe and autonomous satellite navigation. This study builds upon Phase One, which evaluated six key algorithms (RK4, RK8, ODE45, ODE78, ODE113, and MPCI) under fixed step-size constraints and simplified gravitational models. While Phase One provided valuable insights, limitations in adaptability, accuracy over extended durations, and realism of the forces modeled prompted further investigation.
Phase Two introduces significant enhancements, including dynamic step-size recalculation, the integration of Gauss-Lobatto quadrature for improved stability and precision, and the incorporation of additional forces such as atmospheric drag and perturbations into the second equation of motion. The algorithms were implemented and tested under realistic constraints on a virtual machine configured to emulate satellite onboard computer limitations (single-core processor at 500 MHz, <64 MB memory).
Five satellites, representing diverse orbital regimes, were selected for testing, and performance metrics—position differences compared to a baseline and execution times—were analyzed. Results show that RK8 achieves the best balance between accuracy and computational efficiency, while RK4 demonstrates suitability for time-critical tasks with lower precision requirements. The Adaptive Picard-Chebyshev Iteration (APCI), though highly accurate, proved impractical due to its excessive execution time.
This study highlights the trade-offs between accuracy and computational efficiency for various state propagation algorithms, providing critical insights for real-time collision avoidance and autonomous satellite navigation. Future work includes further optimization of APCI and extending testing to real CubeSat environments.

Key words:
Satellite, Space debris, State propagation, Runge-Kutta, ODE solvers, Numerical methods, Modified Picard-Chebyshev Iteration (MPCI), Satellite navigation, Orbit prediction, SGP4 model, Algorithm evaluation, 
Position and velocity approximation, Autonomous satellites, Real-time computational efficiency, Adaptive Picard-Chebyshev, gauss lobatto.

note: the folder named Project contains phase one of the project , the folder named Project-phase2 contains the second phase of the project 
