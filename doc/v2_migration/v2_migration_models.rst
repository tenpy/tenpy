.. todo ::
    This is still a draft!

# Migrate a custom model based on CouplingMPOModel:

* replace base class with ModelBuilder
* dont need to subclass NearestNeighbourModel
* Update init_sites: can probably mostly stay the same, except return lasyten sites, should probably do self.conserve = conserve for convenience
* implement init_couplings, see e.g. SpinModel
* Update init_terms: Use new methods with different signatures, strengths can stay the same
