{{ name | escape | underline }}

- full name: {{ fullname | escape }}
- parent module: :mod:`{{ module }}`
- type: {{ objtype }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
    :members:
    :member-order: bysource
    :inherited-members:
    :show-inheritance:
