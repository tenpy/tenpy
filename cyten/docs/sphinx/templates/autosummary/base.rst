{{ name | escape | underline}}

- full name: {{ fullname | escape }}
- parent module: :mod:`{{ module }}`
- type: {{ objtype }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
