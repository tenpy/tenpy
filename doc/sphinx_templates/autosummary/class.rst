{{ name | escape | underline }}

- full name: {{ fullname | escape }}
- parent module: :mod:`{{ module }}`
- type: {{ objtype }}

{% if methods %}
.. rubric:: Methods

.. autosummary::
{% for item in methods %}
    ~{{ fullname }}.{{ item }}
{%- endfor %}
{% endif %}
{% if attributes %}
.. rubric:: Class Attributes and Properties

.. autosummary::
{% for item in attributes %}
    ~{{ fullname }}.{{ item }}
{%- endfor %}
{% endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ fullname }}
    :members:
    :member-order: bysource
    :inherited-members:
    :show-inheritance:
