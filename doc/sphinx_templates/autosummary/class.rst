{{ name | escape | underline }}

- full name: {{ fullname | escape }}
- parent module: :mod:`{{ module }}`
- type: {{ objtype }}

.. currentmodule:: {{ module }}

.. rubric:: Inheritance Diagram

.. inheritance-diagram:: {{ fullname }}
    :parts: 1

{% if methods %}
.. rubric:: Methods

.. autosummary::
{% for item in methods %}
    {{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% if attributes %}
.. rubric:: Class Attributes and Properties

.. autosummary::
{% for item in attributes %}
    {{ name }}.{{ item }}
{%- endfor %}
{% endif %}


.. autoclass:: {{ fullname }}
    :members:
    :member-order: bysource
    :inherited-members:
    :show-inheritance:
