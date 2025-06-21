

.. _sphx_glr_auto_examples_unsteady:

.. _mfv2d.examples.unsteady:

Examples Unsteady
=================

Examples of the code.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cavity flow can be considered a canonical solution for the Navier-Stokes, given how well the solution to this problem is known. In this exampled it is solved for the case of Re = 10, since that allows for quick convergence on a fairly coarse grid.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_cavity_flow_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_cavity_flow.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Navier-Stokes: Cavity Flow</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The (unsteady) heat equation is essentially just the Poisson equation with a time derivative term added to the left side, as per equation unsteady-heat-direct-equation.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_heat_direct_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_heat_direct.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Unsteady Heat Equations in Direct Formulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Vector reaction equation examples is solving the same equations as sphx_glr_auto_examples_unsteady_plot_reaction.py and sphx_glr_auto_examples_unsteady_plot_reaction_mixed.py, but with the u being a 1-form. This does just makes the solution have two decoupled components, which are solved for independently.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_vector_reaction_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_vector_reaction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Vector Reaction Equation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example is exactly the same as sphx_glr_auto_examples_unsteady_plot_heat_direct.py. As such, only differences from that one will be mentioned.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_heat_mixed_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_heat_mixed.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Unsteady Heat Equations in Mixed Formulation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Mixed reaction equation examples is exactly the same equation as the one in sphx_glr_auto_examples_unsteady_plot_reaction.py, but with the difference being that now u is taken to be a 2-form instead of a 0-form. As such, there is now a real necessety to introduce a 1-form for its derivative.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_reaction_mixed_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_reaction_mixed.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mixed Reaction Equation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The reaction equation is probably the simples non-trivial unsteady differenital equation. It is given by equation unsteady-reaction-direct-equation. If the forcing f is only space dependent, then it has a very simple analytical solution given by unsteady-reaction-direct-solution, where u_\mathrm{initial} = u(x, y, 0) and u_\mathrm{final} = f(x, y) / \alpha.">

.. only:: html

  .. image:: /auto_examples/unsteady/images/thumb/sphx_glr_plot_reaction_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_unsteady_plot_reaction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reaction Equation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/unsteady/plot_cavity_flow
   /auto_examples/unsteady/plot_heat_direct
   /auto_examples/unsteady/plot_vector_reaction
   /auto_examples/unsteady/plot_heat_mixed
   /auto_examples/unsteady/plot_reaction_mixed
   /auto_examples/unsteady/plot_reaction

