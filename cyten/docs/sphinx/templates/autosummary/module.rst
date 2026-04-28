{{ name | escape | underline }}

- full name: {{ fullname | escape }}
- parent module: :mod:`{{ module }}`
- type: {{ objtype }}

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. inheritance-diagram:: {{ fullname }}
    :parts: 1

.. autosummary::
    :toctree: .
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
    :toctree: .
    {% for exc in exceptions %}
    {{ exc}}
    {% endfor %}

{% endif %}{% if functions %}
.. rubric:: Functions

.. autosummary::
    :toctree: .
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}

.. rubric:: Module description

.. automodule:: {{ fullname }}
    :show-inheritance:
