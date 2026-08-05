OPEN ACCESS 

**RESEARCH ARTICLE** APPLIED PHYSICAL SCIENCES 









# **Deep reinforcement learning control unlocks enhanced heat transfer in turbulent convection** 

Zisong Zhou<sup>a</sup> ID and Xiaojue Zhu<sup>a,1</sup> ID 

Edited by David Weitz, Harvard University, Cambridge, MA; received March 20, 2025; accepted August 10, 2025 

**Turbulent convection governs heat transport in both natural and industrial settings, yet optimizing it under extreme conditions remains a significant challenge. Traditional control strategies, such as predefined temperature modulation, struggle to achieve substantial enhancement. Here, we introduce a deep reinforcement learning (DRL) framework that autonomously discovers optimal control policies to maximize heat transfer in turbulent Rayleigh-Bénard convection. By dynamically adjusting wall temperature fluctuations, the DRL agent achieves a heat transfer enhancement of up to 38.5%, exceeding the 20 to 25% limit of conventional methods. The learned strategy reveals a nonlinear state–action relationship, inducing a fully modulated boundary layer regime. Furthermore, we distill the DRL insights into a simplified bang-bang control model, which retains comparable performance (up to 40.0% enhancement) and, crucially, generalizes to unseen, higher Rayleigh number cases without additional training. Our results demonstrate the power of machine learning in turbulence control and reveal a framework with potential for intelligent heat transfer optimization in real-world applications.** 

### turbulent convection | reinforcement learning | turbulence control 

Turbulent convection plays a fundamental role in shaping both natural phenomena and engineered systems, driving energy transport across a vast range of scales. In nature, buoyancy-driven turbulent convection governs atmospheric dynamics, influencing weather patterns and global climate systems (1), while also sustaining ocean currents that regulate thermohaline circulation (2). It underlies key geophysical processes, such as mantle convection that shapes Earth’s interior (3), and astrophysical phenomena, including stellar energy transport and magnetic field generation (4, 5). In technology, turbulent convection is central to optimizing heat exchangers (6), enhancing thermal management in electronic cooling, and improving industrial processes such as semiconductor crystal growth (7) and advanced thermal stabilization (8). Despite its ubiquity, turbulent convection remains a challenge to control and predict, motivating decades of research into its fundamental mechanisms. 

An archetypal model for investigating turbulent convection is Rayleigh-Bénard (RB) convection, where a horizontal fluid layer of height _H_ is uniformly heated from below at temperature _Tb_ and cooled from above at temperature _Tt_ , resulting in a temperature difference _Tb_ − _Tt_ = Δ _T_ . This simplified system serves as a foundational framework for exploring convection and its relevance to both natural phenomena and engineering applications. The heat transfer properties of RB convection are defined by three key dimensionless numbers: the Rayleigh number ( _Ra_ ), the Prandtl number ( _Pr_ ), and the Nusselt number ( _Nu_ ) (4, 5, 9, 10). _Ra_ quantifies the intensity of thermal driving and is expressed as _Ra_ = _훼g_ Δ _TH_<sup>3</sup> _/_ ( _휈휅_ ), where _훼_ is the thermal expansion coefficient, _g_ the gravitational acceleration, _휈_ the kinematic viscosity, and _휅_ the thermal diffusivity. Meanwhile, _Pr_ characterizes the fluid’s momentum-to-thermal diffusivity ratio through _Pr_ = _휈/휅_ . In contrast, the Nusselt number ( _Nu_ ) measures the efficiency of heat transfer relative to pure conduction, defined as _Nu_ = _Q/_ ( _휆_ Δ _T /H_ ), where _Q_ is the total heat flux and _휆_ is the thermal conductivity. _Nu_ reflects the enhancement of heat transport due to convection and serves as a critical metric for evaluating thermal performance in 

In recent years, various strategies have been employed to control heat transfer in RB convection, with a particular focus on modifying flow structures and boundary layers. These approaches include surface roughness (11, 12), time-varying heating (13), spatial temperature modulation (8, 14), and the application of external forces such as vibrations (7), magnetic fields (15), and the addition of particles (16). For example, spatially harmonic heating has been shown to affect the emission of thermal plumes and the large-scale circulation, thereby enhancing heat transfer by modulating convective dynamics (14). 

## **Significance** 

Enhancing heat transfer in turbulent flows is vital for energy systems and industrial processes, yet conventional methods yield limited gains. We demonstrate how artificial intelligence autonomously discovers superior control strategies. A deep learning system dynamically adjusts thermal boundaries in turbulence simulations, achieving 38.5% heat transfer enhancement—over 50% better than traditional approaches. The AI-derived strategy is distilled into a simple formula that retains effectiveness even in extreme, unencountered conditions without additional computations. This breakthrough contributes to bridging the gap between data-driven control and real-world applications, offering a framework for advanced turbulent flow control. 

Author affiliations:<sup>a</sup> Max Planck Institute for Solar System Research, Göttingen 37077, Germany 

Author contributions: Z.Z. and X.Z. designed research; performed research; contributed new reagents/analytic tools; analyzed data; and wrote the paper. The authors declare no competing interest. This article is a PNAS Direct Submission. 

Copyright © 2025 the Author(s). Published by PNAS. This open access article is distributed under Creative Commons Attribution License 4.0 (CC BY). 1To whom correspondence may be addressed. Email: zhux@mps.mpg.de. 

This article contains supporting information online at https://www.pnas.org/lookup/suppl/doi:10.1073/pnas. 2506351122/-/DCSupplemental. 

Published September 9, 2025. 

https://doi.org/10.1073/pnas.2506351122 **1 of 7** 

**PNAS** 2025 Vol. 122 No. 37 e2506351122 

Low-wavenumber wall temperature fluctuations, with a sinusoidal distribution along the horizontal direction, can notably increase _Nu_ , with a maximum heat transfer enhancement of approximately 25% observed in three-dimensional simulations (14). However, despite these advances, existing methods often rely on predefined control strategies, which limit their optimization potential. It remains unclear whether current control strategies can be improved to achieve greater heat transfer enhancement, highlighting the need for further investigation. 

To address the optimization challenges in flow control techniques for RB convection, the application of DRL emerges as a promising solution. DRL has garnered significant attention in various domains, including video classification (17) and speech recognition (18), due to its ability to autonomously optimize complex decision-making tasks. In fluid mechanics, DRL is increasingly utilized for flow control problems (19–33), offering an innovative shift from conventional methods that often rely heavily on human insights. By leveraging neural networks, DRL can build nonlinear models between inputs and controls, potentially uncovering adaptive strategies that are more finely attuned to the complex dynamics of turbulence (34, 35), potentially leading to substantial improvements in control strategy performance. While DRL has shown promise in stabilizing flow and convection patterns at relatively low _Ra_ (22, 29, 33), controlling RB convection at high _Ra_ (not lower than 10<sup>7</sup> ), where turbulence becomes pronounced (36), remains a significant challenge. This high- _Ra_ regime is more commonly encountered in engineering applications, where efficient heat transfer enhancement is crucial. 

In this work, we deploy DRL method to optimize heat transfer in high- _Ra_ RB convection. While existing strategies rely on predefined actuation patterns, our approach enables autonomous discovery of nonlinear control policies that dynamically adapt to turbulent convection. We demonstrate that DRL-driven wall temperature modulation achieves heat transfer enhancement up to 38.5% at _Ra_ = 5×10<sup>8</sup> for _Pr_ = 1, significantly outperforming traditional methods. This improvement is attributed to the emergence of a fully modulated boundary layer regime, where thermal perturbations penetrate deeply into the central flow region. Notably, the learned strategy can be distilled into a simplified threshold-based model without training process, retaining comparable performance while reducing computational complexity, which even achieves 40.0% heat transfer enhancement under higher _Ra_ = 1 × 10<sup>9</sup> . 

## **Results** 

**DRL-Based Control Strategies.** In this study, heat transfer enhancement is achieved through the implementation of nonuniform heating on the lower wall. During the control process, the average temperature _Tb_ of the lower wall _x_ = 0 remains constant, while spatially and temporally varying temperature fluctuations _Tw_<sup>′(</sup><sup>_y, z, t_)serveascontrolinputs.Here,(</sup><sup>_x, y, z_)representsthe</sup> coordinates in the wall-normal direction and the two orthogonal horizontal directions, respectively, with _t_ denoting time. 

To optimize turbulent flow control, we employ DRL to govern the temperature fluctuations _Tw_<sup>′,whichconstitutethe</sup> control actions _at_ . The control system consists of two core components: the numerical simulation and the DRL agent, as illustrated in Fig. 1. The simulation serves as the environment, providing both the instantaneous flow state _st_ and the reward _rt_ that quantifies system performance. The DRL agent processes these inputs, iteratively updates its decision-making policy _휋_ ( _st_ ), 













**Fig. 1.** The flow chart of DRL-based control in turbulent convection. The numerical simulation environment provides state information and reward to the DRL agent, which optimizes its policy through actor–critic network and outputs the next action to control the flow field. 

and generates actuation commands _at_ to manipulate the flow field. This closed-loop architecture enables adaptive flow control, with the agent continuously refining its policy through realtime interaction with the flow system. Within this framework, the wall temperature fluctuations _Tw_<sup>′aredesignatedascontrol</sup> actions _at_ , with amplitude constrained to the range [−Δ _T,_ Δ _T_ ]. Meanwhile, temperature fluctuations _T휆_<sup>′at the estimated thermal</sup> boundary layer height _휆휃_ = 1 _/_ (2 _Nu_ 0) near the lower wall are selected as the states _st_ , where _Nu_ 0 denotes the baseline _Nu_ in uncontrolled flow. The reward function _rt_ is formulated as _휂_ , the relative enhancement of _Nu_ compared to the baseline value, calculated from the mean heat flux on the lower wall. Through policy gradient optimization, the agent progressively enhances its control strategy to maximize the cumulative reward, thereby driving systematic performance improvement. Further details of the methodology are provided in _Materials and Methods_ . 

The heat transfer enhancement achieved through DRL-based control strategies is quantitatively characterized by the elevated Nusselt number _Nud_ and enhancement ratio _휂d_ shown in Table 1, while all results have been statistically convergent as described in _SI Appendix_ . These strategies demonstrate sustained effectiveness at _Ra_ up to 5 × 10<sup>8</sup> , with all cases exhibiting _휂d_ exceeding 30%, peaking at 38.5% when _Ra_ = 5 × 10<sup>8</sup> . This performance significantly surpasses the maximum 20% to 25% enhancement ratio attainable through predefined sinusoidal temperature fluctuation controls (14). 

**Mechanism of Heat Transfer Enhancement.** Further investigations are warranted to elucidate the impact of DRL-based control on heat transfer and their underlying flow modification mechanisms. First, systematic analysis of the correlation between DRL-generated actuation signals and their corresponding input states is crucial, with their spatial distributions captured in Fig. 2 _A_ and _B_ . These instantaneous fields correspond to the flow state at 1,000 free-fall time units ( _t_ 0 = � _H /_ ( _훼g_ Δ _T_ )) after DRL-based control implementation, when the turbulent flow has been fully developed. The output wall temperature fluctuations _Tw_<sup>′generallyexhibitthesamesignastheinput</sup> signal _T휆_<sup>′atthecorrespondinghorizontallocations.Thisphase</sup> alignment is accompanied by significant amplitude amplification, with distinct localized hot phase regions reaching the prescribed temperature actuation limit, Δ _T_ . Notably, these characteristic hot phase regions persist while thermal structures become increasingly fragmented with elevated _Ra_ , and the boundaries 

**2 of 7** https://doi.org/10.1073/pnas.2506351122 

pnas.org 



<!-- Start of picture text -->
SealCr o~ tw NeCr o~ tw Ne o~ tw Ne tw Ne Ne <-- 0.5nn<br>I = — 30%<br>g, o5 | ! : 20%<br>”%<br>- 1 - 0.5 0 0.5 1 - l - 0.5 0 0.5 1 - l - 0.5 0 0.5 1<br>TAT TAT TAT<br><!-- End of picture text -->



<!-- Start of picture text -->
SealCr o~ tw NeCr o~ tw Ne o~ tw Ne tw Ne Ne <-- 0.5nn<br>I = — 30%<br>g, o5 | ! : 20%<br>”%<br>- 1 - 0.5 0 0.5 1 - l - 0.5 0 0.5 1 - l - 0.5 0 0.5 1<br>TAT TAT TAT<br><!-- End of picture text -->



<!-- Start of picture text -->
I = — 30%<br>g, o5 | ! : 20%<br>”%<br>- 1 - 0.5 0 0.5 1 - l - 0.5 0 0.5 1 - l - 0.5 0 0.5 1<br>TAT TAT TAT<br><!-- End of picture text -->



<!-- Start of picture text -->
<_<br>fa ail,<br><!-- End of picture text -->



<!-- Start of picture text -->
; a @ >.<br>‘O<br><!-- End of picture text -->



<!-- Start of picture text -->
: i ee<br>5<br><!-- End of picture text -->



<!-- Start of picture text -->
Ss ae ey<br>6<br><!-- End of picture text -->



<!-- Start of picture text -->
; a @ >. = : i ee Ss ae ey<br>‘O 5 6<br><_<br>fa ail, Tw Cold phase Hot phase<br>Case 1E7  =35.8% > =<br>801<br>0 . wee<br>F 10 a02 7 =<br>| a, 0 nee<br>7 Pa WyxX 0.5 > = 1<br>~=—= —_S— ~<a ‘. : \ 0 1 2<br>Case=1E =30.29 } E =38.5% iA 5 0.5, 1 152 ,<br>1E8 = 4=30.2% Case 5SE8 = 7=38.5% ((L) —T,)/AT<br><!-- End of picture text -->



<!-- Start of picture text -->
F<br>7 Pa<br>~=—= —_S—<br><!-- End of picture text -->



<!-- Start of picture text -->
«x :<br><4924 )<br>State. 1<br>By<br>WN<br>| Looo<br>“ 1<br><!-- End of picture text -->



<!-- Start of picture text -->
:<br>)<br>.<br>/ 5<br>Li<br>State. 1 ye _ : *# | A q<br>> <. a =<br>By : Se<br>WN<br>| Looo Case 1E7 7=35.1% Case IES 7=37.8%<br>“ 1<br>- 1 0 1<br>Ti/AT/AT 10<br>3<br>,resress y = aKA = aKA aKAA _ 4 34 le re ’ 4 .p 0.5<br> » far Ss < — 0.0<br>> >p32p3232 = — ts . .<br>= Case SES —17=36.8% Case 1E9 =40.0% VAT<br><!-- End of picture text -->



<!-- Start of picture text -->
By<br>WN<br>| Looo<br>“ 1<br>Action - 1 0 1<br>Ti/AT/AT<br>— ee”<br> SF : 4 ,resress y = aKA = aKA aKAA<br>‘ 4° » far<br>% ne> >p32p3232<br>=<br><!-- End of picture text -->

## **Discussion and Conclusion** 

Our study demonstrates that DRL offers an innovative framework for optimizing heat transfer enhancement in high- _Ra_ turbulent RB convection, achieving notable performance beyond conventional control strategies. By leveraging adaptive, nonlinear control policies, DRL-based thermal actuation generates sustained heat transfer improvements exceeding 30% at Rayleigh numbers up to _Ra_ = 5 × 10<sup>8</sup> , significantly surpassing the 20 to 25% enhancement limits of predefined sinusoidal modulation methods. 

A key finding lies in the distinct nonlinear state–action relationship governing the DRL-generated control strategy. Positive thermal fluctuations at the boundary layer trigger maximal heating +Δ _T_ , while negative inputs yield milder cooling responses above −Δ _T_ . This nonlinearity enables the system to amplify hot plume ejections while maintaining stable temperature inversion over cold-phase regions, critical for enhancing turbulent convection. The resulting fully modulated boundary layer regime, characterized by penetration and inversion depths exceeding 10 _휆휃_ , confirms that DRL-driven actuation penetrates into the central flow region, leading to a significant increase in heat transfer. 

The development of a simplified bang-bang control strategy, inspired by DRL insights, further underscores the robustness and scalability of this approach. By translating the learned nonlinear policy into a threshold-based hyperbolic tangent function, which constitutes a bang-bang controllerwithout hysteresis, we achieved comparable heat transfer enhancements (e.g., 40.0% at _Ra_ = 10<sup>9</sup> ) without requiring iterative DRL training. This simplification retains the core mechanism of phase-aligned amplification while ensuring computational feasibility for further industrial applications. While sinusoidal controllers are widely studied for their mathematical convenience in linearized flow analysis, our work shows that bang-bang controllers-often distilled from DRL policies-are effective alternatives for turbulent flow control. These controllers represent a fundamental class of simple control laws deserving increased theoretical attention. This breakthrough also highlights the potential of machine learning to uncover control mechanisms that are finely attuned to the dynamics of turbulent flows, particularly in regimes where traditional approaches struggle to balance complexity and efficiency. 

## **Materials and Methods** 

**Deep Reinforcement Learning.** The DRL framework illustrated in Fig. 1 employs the twin-delayed deep deterministic policy gradient (TD3) algorithm (43), utilizing an open-source implementation tailored for turbulence control by Lee et al. (30). Both input states and output actions are represented as twodimensional arrays with dimensions _Ny_ × _Nz_ , while the instantaneous reward corresponds to the relative _Nu_ enhancement. The TD3 model is an actor–critic network architecture designed to enhance learning stability and performance by addressing overestimation bias, employing delayed updates, and applying target smoothing techniques. During the training process, the TD3 algorithm aims to optimize the action value function _q휋_ ( _st, at_ ) through satisfaction of the Bellman equation: 



where _rt_<sup>_d_=�</sup><sup>_n_</sup> _j_ =1<sup>_훾j_−1</sup><sup>_rt_+</sup><sup>_j_denotesthe</sup><sup>_n_-stepdiscountedreward,</sup><sup>_훾_the</sup> discountfactor, _휋휙_ ( _st_ + _n_ ) thedelayedpolicyupdate,and _휖_ theclippedrandom noise. Parameters are set to _n_ = 5 and _훾_ = 0 _._ 95 following the open-source code. The critic networks estimate the expected cumulative reward, with their 

**Table 2** . **Number of samples used for the statistics** 

|Cases|_ny_|_nz_|_nt_|_n_sum|_t_sum_/t_0|
|---|---|---|---|---|---|
|1E7|256|256|150|9_._8×10<sup>6</sup>|1_,_500|
|1E8|384|384|150|2_._2×10<sup>7</sup>|1_,_500|
|5E8|480|480|100|2_._3×10<sup>7</sup>|1_,_000|



We display the number of samples for each case: _ny_ , _nz_ , and _nt_ denote sample numbers in the _y_ , _z_ , and _t_ directions, and _n_ sum = _ny nz nt_ is the total number of samples. _t_ sum is the sampling time period. 

parameters updated via the objective function: 



where _N_ = 64 represents the minibatch size, and _휃_ is the critic network weights. The actor network seeks to determine an optimal policy, guided by the policy objective function _J_ ( _휙_ ), with _휙_ denoting the weight parameters of either network. The objective function is ultimately optimized through: 



where _휓_ denotes the actor network weights. 

ThenetworkarchitecturesintheDRLframeworkarestructuredasfollows(30): The actor network consists of three convolutional layers, which progressively reduce feature complexity by employing 64, 32, and 1 filters respectively, with each filter kernel sized at 3 × 3. In contrast, the critic network adopts a deeper architecturecomprisingsixconvolutionallayersfollowedbythreefullyconnected layers. Each convolutional layer contains 32 filters of size 3 × 3. To enhance feature abstraction, an average pooling layer is strategically inserted after every pair of convolutional layers. The subsequent fully connected layers maintain dimensionalconsistencywith32neuronsperlayer,ultimatelyproducingascalar _q_ -value output to assess the quality of the control policy. The ReLU activation function is applied uniformly across all layers in both networks. 

The training process requires balancing temporal development of control strategy adjustments with computational efficiency. We establish each state step duration as five free-fall time units (5 _t_ 0) to permit sufficient flow response, while limiting episodes to 100 _t_ 0 (20 sequential state steps) to constrain computational costs. This configuration implements policy updates every 5 steps, ensuring alignment between gradient updates and the _n_ -step reward horizon. During the training phase of the control strategy, we formulate the normalizedreward _<u>r</u>_ as _<u>r</u>_ =<sup>�</sup><sup>_n_</sup> _j_ =1<sup>_훾j_−1</sup><sup>_rt_+</sup><sup>_j/_�</sup><sup>_n_</sup> _j_ =1<sup>_훾j_−1.Thismetricservesto</sup> assessthealgorithm’slearningperformance.Oscillationsin _<u>r</u>_ aroundlow-reward value primarily occur within the initial 10 training episodes. Subsequently, the normalized reward rapidly increases and reaches a high-reward plateau, with the metric stabilizing to demonstrate convergence of the DRL models, also documented in _SI Appendix_ . Across all cases, the maximum _<u>r</u>_ values appear within the high-reward plateau range of 10 to 25 episodes. As suggested by Leeetal.(30),extendedtrainingperiodscanresultinissuessuchascatastrophic forgetting or overfitting, which could potentially lead to training failure. Thus, the final control strategy is chosen based on the episode achieving the peak _<u>r</u>_ value. 

Further implementation details of the DRL framework can be found in _SI Appendix_ and Lee et al. (30). In addition, the code for the DRL-based control is shared via GitHub at https://github.com/zhouzisong1997/DRL_RB_control (44). 

**Direct Numerical Simulations.** We consider RB convection established between two parallel plates, with the fluid being heated from below and cooled fromabove.Thegoverningequationsarethethree-dimensionalincompressible Navier–Stokes equations within the Boussinesq approximation, written as 



**6 of 7** https://doi.org/10.1073/pnas.2506351122 

pnas.org 

_∂T ∂t_<sup>+</sup><sup>_u_· ∇</sup><sup>_T_=</sup><sup>_휅_∇2</sup><sup>_T,_</sup> [8] 

where _u_ isthevelocityvector, _P_ isthekinematicpressure,and _x_ ˆ isthewall-normal unit vector. The flow is assumed to be periodic in the horizontal directions. Noslip and no-penetration boundary conditions are imposed on the two walls, where the velocity is set to _u_ = 0. The top wall is maintained at a constant temperature _Tt_ , while the bottom wall is subjected to temperature modulation with a temperature of _Tb_ + _Tw_<sup>′</sup> ( _y, z, t_ ). 

Direct numerical simulations were performed using the AFiD code (45–47). The numerical method employs an energy-conserving, second-order finite difference scheme for spatial discretization, with velocities on a staggered grid. Time marching is achieved through a third-order Runge–Kutta scheme, complemented by a Crank-Nicolson method for implicit terms. The computational grid features uniform spacing in the horizontal directions, while employing clipped Chebyshev-typeclusteringinthewall-normaldirectiontoresolveboundarylayer dynamics near the plates. This code has been extensively validated in previous RB convection studies (45–50). 

Inthisstudy,allthecasesdiscussedhavereachedafullydevelopedturbulent statewithin500 _t_ 0 followingcontrolimplementation.Consequently,allstatistical averaging commences from 500 _t_ 0 after the control is applied. To account for the fluctuations in _Nu_ measurements, a temporal averaging window of 1000 _t_ 0 

1. D. L. Hartmann, L. A. Moy, Q. Fu, Tropical convection and the energy balance at the top of the atmosphere. _J. Clim._ 14, 4495–4511 (2001). 

2. E. van Doorn, B. Dhruva, K. R. Sreenivasan, V. Cassella, Statistics of wind direction and its increments. _Phys. Fluids_ 12, 1529–1534 (2000). 

3. G. A. Glatzmaiers, P. H. Roberts, A three-dimensional self-consistent computer simulation of a geomagnetic field reversal. _Nature_ 377, 203–209 (1995). 

4. G. Ahlers, S. Grossmann, D. Lohse, Heat transfer and large scale dynamics in turbulent RayleighBénard convection. _Rev. Mod. Phys._ 81, 503–537 (2009). 

5. D. Lohse, K. Q. Xia, Small-scale properties of turbulent Rayleigh-Bénard convection. _Annu. Rev. Fluid Mech._ 42, 335–364 (2010). 

6. S. Grossmann, D. Lohse, Scaling in thermal convection: A unifying theory. _J. Fluid Mech._ 407, 27–56 (2000). 

7. B. F. Wang, Q. Zhou, C. Sun, Vibration-induced boundary-layer destabilization achieves massive heat-transport enhancement. _Sci. Adv._ 6, eaaz8239 (2020). 

8. S. Zhang _et al_ ., Stabilizing/destabilizing the large-scale circulation in turbulent Rayleigh-Bénard convection with sidewall temperature control. _J. Fluid Mech._ 915, A14 (2021). 

9. F. Chillà, J. Schumacher, New perspectives in turbulent Rayleigh-Bénard convection. _Eur. Phys. J. E_ . 35, 1–25.546 (2012). 

10. D. Lohse, O. Shishkina, Ultimate Rayleigh-Bénard turbulence. _Rev. Mod. Phys_ . 96, 035001 (2024). 

11. Y. B. Du, P. Tong, Enhanced heat transport in turbulent convection over a rough surface. _Phys. Rev. Lett._ 81, 987 (1998). 

12. H. Jiang _et al_ ., Controlling heat transport and flow structures in thermal turbulence using ratchet surfaces. _Phys. Rev. Lett._ 120, 044501 (2018). 

13. R. Yang _et al_ ., Periodically modulated thermal convection. _Phys. Rev. Lett._ 125, 154502 (2020). 

14. C. B. Zhao _et al_ ., Modulation of turbulent Rayleigh-Bénard convection under spatially harmonic heating. _Phys. Rev. E_ 105, 055107 (2022). 

15. Z. Wang, Z. Zhou, External natural convection heat transfer of liquid metal under the influence of the magnetic field. _Int. J. Heat Mass Transf._ 134, 175–184 (2019). 

16. Z. Wang, V. Mathai, C. Sun, Self-sustained biphasic catalytic particle turbulence. _Nat. Commun._ 10, 3333 (2019). 

17. A. Karpathy _et al_ ., “Large-scale video classification with convolutional neural networks” in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (IEEE, 2014), pp. 1725–1732. 

18. G. Hinton _et al_ ., Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. _IEEE Signal Process. Mag._ 29, 82–97 (2012). 

19. F. Guéniat, L. Mathelin, M. Y. Hussaini, A statistical learning strategy for closed-loop control of fluid flows. _Theoret. Comput. Fluid Dyn._ 30, 497–510 (2016). 

20. J. Rabault, M. Kuchta, A. Jensen, U. Réglade, N. Cerardi, Artificial neural networks trained through deep reinforcement learning discover control strategies for active flow control. _J. Fluid Mech._ 865, 281–302 (2019). 

21. B. Z. Han, W. X. Huang, Active control for drag reduction of turbulent channel flow based on convolutional neural networks. _Phys. Fluids_ 32, 095108 (2020). 

22. G. Beintema, A. Corbetta, L. Biferale, F. Toschi, Controlling Rayleigh-Bénard convection via reinforcement learning. _J. Turbul._ 21, 585–605 (2020). 

23. S. L. Brunton, B. R. Noack, P. Koumoutsakos, Machine learning for fluid mechanics. _Annu. Rev. Fluid Mech._ 52, 477–508 (2020). 

24. R. Paris, S. Beneddine, J. Dandois, Robust flow control and optimal sensor placement using deep reinforcement learning. _J. Fluid Mech._ 913, A25 (2021). 

25. K. Zeng, M. D. Graham, Symmetry reduction for deep reinforcement learning active control of chaotic spatiotemporal dynamics. _Phys. Rev. E_ 104, 014210 (2021). 

26. J. Li, M. Zhang, Reinforcement-learning-based control of confined cylinder wakes with stability analyses. _J. Fluid Mech._ 932, A44 (2022). 

is applied to obtain each _Nu_ in Table 1, ensuring statistical stationarity with a sample size of 10<sup>5</sup> for each case. The sample numbers for the state–action map (Fig. 2 _C_ ) and temperature conditional averaging (Fig. 3 _G_ – _I_ ) are summarized in Table 2. 

**Data, Materials, and Software Availability.** Code for the DRL-based control has been deposited in GitHub (https://github.com/zhouzisong1997/DRL_RB_ control) (44). All other data are included in the manuscript and/or supporting information. 

**ACKNOWLEDGMENTS.** We gratefully acknowledge the financial support from the Max Planck Society, the German Research Foundation from grants 521319293 and 540422505, 550262949, and the Daimler and Benz Foundation. We also thank the High Performance Computing systems of Max Planck Computing and Data Facility for the allocation of the computational time. We gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this project by providing computing time on theGaussCentreforSupercomputingSupercomputersSuperMUC-NGatLeibniz Supercomputing Centre (www.lrz.de) and JUWELS at Jülich Supercomputing Centre. Open access funding provided by the Max Planck Society. 

27. L. Guastoni, J. Rabault, P. Schlatter, H. Azizpour, R. Vinuesa, Deep reinforcement learning for turbulent drag reduction in channel flows. _Eur. Phys. J. E_ 46, 27 (2023). 

28. T. Sonoda, Z. Liu, T. Itoh, Y. Hasegawa, Reinforcement learning of control strategies for reducing skin friction drag in a fully developed turbulent channel flow. _J. Fluid Mech._ 960, A30 (2023). 

29. C. Vignon _et al_ ., Effective control of two-dimensional Rayleigh-Bénard convection: Invariant multiagent reinforcement learning is all you need. _Phys. Fluids_ 35, 065146 (2023). 

30. T. Lee, J. Kim, C. Lee, Turbulence control for drag reduction through deep reinforcement learning. _Phys. Rev. Fluids_ 8, 024604 (2023). 

31. P. Suárez _et al_ ., Flow control of three-dimensional cylinders transitioning to turbulence via multiagent reinforcement learning. _Comms. Eng._ 4, 113 (2025). 

32. R. Vinuesa, Perspectives on predicting and controlling turbulent flows through deep learning. _Phys. Fluids_ 36, 031401 (2024). 

33. J. Vasanth, J. Rabault, F. Alcántara-Ávila, M. Mortensen, R. Vinuesa, “Multi-agent reinforcement learning for the control of three-dimensional Rayleigh-Bénard convection” in _Flow Turbulence and Combustion_ (Springer, 2024), pp. 1–37. 

34. S. L. Brunton, B. R. Noack, Closed-loop turbulence control: Progress and challenges. _Appl. Mech. Rev._ 67, 050801 (2015). 

35. T. Duriez, S. L. Brunton, B. R. Noack, _Machine Learning Control-Taming Nonlinear Dynamics and Turbulence_ (Springer, 2017), vol. 116. 

36. D. Lohse, O. Shishkina, Ultimate Rayleigh-Bénard turbulence. _Rev. Mod. Phys._ 96, 035001 (2024). 

37. R. J. Stevens, A. Blass, X. Zhu, R. Verzicco, D. Lohse, Turbulent thermal superstructures in RayleighBénard convection. _Phys. Rev. Fluids_ 3, 041501 (2018). 

38. A. Pandey, J. D. Scheel, J. Schumacher, Turbulent superstructures in Rayleigh-Bénard convection. _Nat. Commun._ 9, 2118 (2018). 

39. A. Blass, R. Verzicco, D. Lohse, R. J. Stevens, D. Krug, Flow organisation in laterally unconfined Rayleigh-Bénard turbulence. _J. Fluid Mech._ 906, A26 (2021). 

40. T. Seyde _et al_ ., Is bang-bang control all you need? solving continuous control with Bernoulli policies. _Adv. Neural Inf. Process. Syst._ 34, 27209–27221 (2021). 

41. Q. Wei, Z. Niu, B. Chen, X. Huang, Bang-bang control applied in airfoil roll control with plasma actuators. _J. Aircr._ 50, 670–677 (2013). 

42. C. Vignon, J. Rabault, R. Vinuesa, Recent advances in applying deep reinforcement learning for flow control: Perspectives and future directions. _Phys. Fluids_ 35, 031301 (2023). 

43. S. Fujimoto, H. Hoof, D. Meger, “Addressing function approximation error in actor-critic methods” in _International Conference on Machine Learning_ , J. Dy, A. Krause, Eds. (PMLR, 2018), pp. 1587–1596. 

44. Z. Zhou, DRL_RB_control. GitHub. https://github.com/zhouzisong1997/DRL_RB_control. Deposited 8 June 2025. 

45. R. Verzicco, P. Orlandi, A finite-difference scheme for three-dimensional incompressible flows in cylindrical coordinates. _J. Comput. Phys._ 123, 402–414 (1996). 

46. E. P. Van Der Poel, R. Ostilla-Mónico, J. Donners, R. Verzicco, A pencil distributed finite difference code for strongly turbulent wall-bounded flows. _Comput. Fluids_ 116, 10–16 (2015). 

47. X. Zhu _et al_ ., AFiD-GPU: A versatile Navier-Stokes solver for wall-bounded turbulent flows on GPU clusters. _Comput. Phys. Commun._ 229, 199–210 (2018). 

48. Jülich Supercomputing Centre, JUWELS cluster and booster: Exascale pathfinder with modular supercomputing architecture at Juelich supercomputing centre. _J. Large Scale Res. Facil_ . 7, e183 (2021). 

49. R. J. Stevens, R. Verzicco, D. Lohse, Radial boundary layer structure and Nusselt number in Rayleigh-Bénard convection. _J. Fluid Mech._ 643, 495–507 (2010). 

50. R. J. Stevens, D. Lohse, R. Verzicco, Prandtl and Rayleigh number dependence of heat transport in high Rayleigh number thermal convection. _J. Fluid Mech._ 688, 31–43 (2011). 

https://doi.org/10.1073/pnas.2506351122 **7 of 7** 

**PNAS** 2025 Vol. 122 No. 37 e2506351122 

