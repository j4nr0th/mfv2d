//
// Created by jan on 7/25/25.
//

#include "forms.h"

static const char *form_order_str_table[] = {
    [FORM_ORDER_UNKNOWN] = "FORM_ORDER_UNKNOWN",
    [FORM_ORDER_0] = "FORM_ORDER_0",
    [FORM_ORDER_1] = "FORM_ORDER_1",
    [FORM_ORDER_2] = "FORM_ORDER_2",
};

const char *form_order_str(form_order_t order)
{
    // ALWAYS RANGE CHECK!
    if (order < FORM_ORDER_UNKNOWN || order > FORM_ORDER_2)
        return "FORM_ORDER_INVALID";
    return form_order_str_table[order];
}
unsigned form_degrees_of_freedom_count(const form_order_t form, const unsigned order_1, const unsigned order_2)
{
    switch (form)
    {
    case FORM_ORDER_0:
        return (order_1 + 1) * (order_2 + 1);
    case FORM_ORDER_1:
        return order_1 * (order_2 + 1) + (order_1 + 1) * order_2;
    case FORM_ORDER_2:
        return order_1 * order_2;
    default:
        return 0;
    }
}

form_order_t form_order_from_object(PyObject *object)
{
    const long val = PyLong_AsLong(object);
    if (PyErr_Occurred() || val < FORM_ORDER_0 || val > FORM_ORDER_2)
    {
        raise_exception_from_current(PyExc_ValueError, "Invalid form order: %ld", val);
        return FORM_ORDER_UNKNOWN;
    }
    return (form_order_t)val;
}

unsigned element_form_offset(const element_form_spec_t *const spec, const unsigned index, const unsigned order_1,
                             const unsigned order_2)
{
    ASSERT(index < Py_SIZE(spec), "Form index %u is out of bounds for an element form spec with %u entries.", index,
           (unsigned)Py_SIZE(spec));

    unsigned offset = 0;
    for (unsigned i = 0; i < index; ++i)
    {
        const form_order_t order = spec->forms[i].order;
        offset += form_degrees_of_freedom_count(order, order_1, order_2);
    }
    return offset;
}

unsigned element_form_specs_total_count(const element_form_spec_t *const spec, const unsigned order_1,
                                        const unsigned order_2)
{
    ASSERT(order_1 > 0 && order_2 > 0, "Orders must be positive, but (%u, %u) were given", order_1, order_2);
    unsigned count = 0;
    for (unsigned i = 0; i < Py_SIZE(spec); ++i)
    {
        count += form_degrees_of_freedom_count(spec->forms[i].order, order_1, order_2);
    }
    return count;
}

typedef struct
{
    PyObject_HEAD;
    element_form_spec_t *efs;
    Py_ssize_t index;
} element_form_spec_iter_t;

static void element_form_spec_iter_dealloc(element_form_spec_iter_t *it)
{
    Py_XDECREF(it->efs);
    Py_TYPE(it)->tp_free((PyObject *)it);
}

static PyObject *element_form_spec_iter_next(element_form_spec_iter_t *it)
{
    if (it->index >= Py_SIZE(it->efs))
    {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    const Py_ssize_t i = it->index;
    it->index += 1;
    return Py_BuildValue("si", it->efs->forms[i].name, it->efs->forms[i].order);
}

MFV2D_INTERNAL
PyTypeObject element_form_spec_iter_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d._ElementFormSpecificationIter",
    .tp_basicsize = sizeof(element_form_spec_iter_t),
    .tp_dealloc = (destructor)element_form_spec_iter_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = PyObject_SelfIter, // the iterator's __iter__ returns self
    .tp_iternext = (iternextfunc)element_form_spec_iter_next,
};

static PyObject *element_form_spec_iter(element_form_spec_t *self)
{
    element_form_spec_iter_t *const it = PyObject_New(element_form_spec_iter_t, &element_form_spec_iter_type);
    if (!it)
    {
        return NULL;
    }
    Py_INCREF(self);
    it->efs = self;
    it->index = 0;
    return (PyObject *)it;
}

static PyObject *element_form_spec_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    const Py_ssize_t num_specs = PyTuple_GET_SIZE(args);
    if (num_specs == 0)
    {
        PyErr_SetString(PyExc_TypeError, "at least one spec must be provided");
        return NULL;
    }

    if (kwds != NULL)
    {
        PyErr_SetString(PyExc_TypeError, "_ElementFormSpecification takes no keyword arguments");
        return NULL;
    }

    element_form_spec_t *const this = (element_form_spec_t *)type->tp_alloc(type, num_specs);
    if (!this)
        return NULL;
    Py_SET_SIZE(this, num_specs);

    for (Py_ssize_t i = 0; i < num_specs; ++i)
    {
        PyObject *item = PyTuple_GET_ITEM(args, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 2)
        {
            PyErr_Format(PyExc_TypeError, "Each spec must be a (name, order) tuple, but element %u was instead %R",
                         (unsigned)i, item);
            Py_DECREF(this);
            return NULL;
        }
        PyObject *name_obj = PyTuple_GET_ITEM(item, 0);
        PyObject *order_obj = PyTuple_GET_ITEM(item, 1);

        if (!PyUnicode_Check(name_obj))
        {
            PyErr_SetString(PyExc_TypeError, "Form name must be string");
            Py_DECREF(this);
            return NULL;
        }
        Py_ssize_t name_len;
        const char *name_cstr = PyUnicode_AsUTF8AndSize(name_obj, &name_len);
        if (!name_cstr)
        {
            raise_exception_from_current(PyExc_ValueError, "Failed converting form name to a string for entry %u.",
                                         (unsigned)i);
            Py_DECREF(this);
            return NULL;
        }

        if (name_len > MAXIMUM_FORM_NAME_LENGTH)
        {
            PyErr_Format(PyExc_ValueError, "Form name longer than %d characters", MAXIMUM_FORM_NAME_LENGTH);
            Py_DECREF(this);
            return NULL;
        }

        // Check for duplicates
        for (unsigned j = 0; j < i; ++j)
        {
            if (strcmp(name_cstr, this->forms[j].name) == 0)
            {
                PyErr_Format(PyExc_ValueError, "Duplicate form name %s for entry %u", name_cstr, (unsigned)i);
                Py_DECREF(this);
                return NULL;
            }
        }

        const form_order_t order_val = form_order_from_object(order_obj);
        if (order_val == FORM_ORDER_UNKNOWN)
        {
            raise_exception_from_current(PyExc_ValueError, "Failed converting form order to a number for entry %u.",
                                         (unsigned)i);
            Py_DECREF(this);
            return NULL;
        }

        strncpy(this->forms[i].name, name_cstr, MAXIMUM_FORM_NAME_LENGTH);
        this->forms[i].name[MAXIMUM_FORM_NAME_LENGTH] = '\0'; // Ensure null-termination
        this->forms[i].order = order_val;
    }

    return (PyObject *)this;
}

static void element_form_spec_dealloc(element_form_spec_t *this)
{
    Py_TYPE(this)->tp_free((PyObject *)this);
}

static PyObject *element_form_spec_get_orders(const element_form_spec_t *self, void *Py_UNUSED(closure))
{
    const Py_ssize_t size = Py_SIZE(self);
    PyObject *res = PyTuple_New(size);
    if (!res)
        return NULL;
    for (Py_ssize_t i = 0; i < size; ++i)
    {
        PyObject *val = PyLong_FromLong(self->forms[i].order);
        if (!val)
        {
            Py_DECREF(res);
            return NULL;
        }
        PyTuple_SET_ITEM(res, i, val);
    }
    return res;
}

static PyObject *element_form_spec_get_names(const element_form_spec_t *self, void *Py_UNUSED(closure))
{
    const Py_ssize_t size = Py_SIZE(self);
    PyObject *res = PyTuple_New(size);
    if (!res)
        return NULL;
    for (Py_ssize_t i = 0; i < size; ++i)
    {
        PyObject *val = PyUnicode_FromString(self->forms[i].name);
        if (!val)
        {
            Py_DECREF(res);
            return NULL;
        }
        PyTuple_SET_ITEM(res, i, val);
    }
    return res;
}

static PyObject *element_form_spec_getitem(const element_form_spec_t *self, const Py_ssize_t index)
{
    if (index < 0 || index >= Py_SIZE(self))
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
    return Py_BuildValue("si", self->forms[index].name, self->forms[index].order);
}

static Py_ssize_t element_form_spec_len(const element_form_spec_t *self)
{
    return Py_SIZE(self);
}

static int element_form_spec_contains(const element_form_spec_t *self, PyObject *item)
{
    if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Each spec must be a (name, order) tuple");
        return -1;
    }
    PyObject *name_obj = PyTuple_GET_ITEM(item, 0);
    PyObject *order_obj = PyTuple_GET_ITEM(item, 1);
    if (!PyUnicode_Check(name_obj))
    {
        PyErr_SetString(PyExc_TypeError, "Form name must be string");
        return -1;
    }
    Py_ssize_t name_len;
    const char *name_cstr = PyUnicode_AsUTF8AndSize(name_obj, &name_len);
    if (!name_cstr)
    {
        raise_exception_from_current(PyExc_ValueError, "Failed converting form name to a string.");
        return -1;
    }

    if (name_len > MAXIMUM_FORM_NAME_LENGTH)
    {
        PyErr_Format(PyExc_ValueError, "Form name longer than %d characters", MAXIMUM_FORM_NAME_LENGTH);
        return -1;
    }

    const form_order_t order_val = form_order_from_object(order_obj);
    if (order_val == FORM_ORDER_UNKNOWN)
    {
        raise_exception_from_current(PyExc_ValueError, "Failed converting form order to a number.");
        return -1;
    }

    for (Py_ssize_t i = 0; i < Py_SIZE(self); ++i)
    {
        // Check order first, since it is a cheaper check.
        if (self->forms[i].order == order_val && strcmp(self->forms[i].name, name_cstr) == 0)
        {
            return 1;
        }
    }
    return 0;
}

static PySequenceMethods element_form_spec_sequence_methods = {
    .sq_item = (ssizeargfunc)element_form_spec_getitem,
    .sq_length = (lenfunc)element_form_spec_len,
    .sq_contains = (objobjproc)element_form_spec_contains,
};

static PyGetSetDef element_form_spec_getset[] = {
    {
        .name = "orders",
        .doc = "tuple[int, ...]: Returns a tuple of form orders.",
        .get = (getter)element_form_spec_get_orders,
        .set = NULL,
    },
    {
        .name = "names",
        .doc = "tuple[str, ...]: Returns a tuple of form names.",
        .get = (getter)element_form_spec_get_names,
        .set = NULL,
    },
};

static PyObject *element_form_spec_repr(const element_form_spec_t *self)
{
    const Py_ssize_t size = Py_SIZE(self);
    PyObject *res = PyUnicode_FromString("_ElementFormSpecification(");
    if (!res)
        return NULL;
    for (Py_ssize_t i = 0; i < size; ++i)
    {
        PyObject *val = PyUnicode_FromFormat("%s(%u)", self->forms[i].name, self->forms[i].order);
        if (!val)
        {
            Py_DECREF(res);
            return NULL;
        }
        PyObject *new = PyUnicode_Concat(res, val);
        Py_DECREF(val);
        if (!new)
        {
            Py_DECREF(res);
            return NULL;
        }
        res = new;
    }
    PyObject *new = PyUnicode_Concat(res, PyUnicode_FromString(")"));
    Py_DECREF(res);
    return new;
}

static PyObject *element_form_spec_str(const element_form_spec_t *self)
{
    const Py_ssize_t size = Py_SIZE(self);
    PyObject *res = PyUnicode_FromString("(");
    if (!res)
        return NULL;
    PyObject *val = PyUnicode_FromFormat("%s(%u),", self->forms[0].name, self->forms[0].order - 1);
    if (!val)
    {
        Py_DECREF(res);
    }
    for (Py_ssize_t i = 1; i < size; ++i)
    {
        val = PyUnicode_FromFormat(", %s(%u),", self->forms[i].name, self->forms[i].order - 1);
        if (!val)
        {
            Py_DECREF(res);
        }
    }
    PyObject *new = PyUnicode_Concat(res, PyUnicode_FromString(")"));
    Py_DECREF(res);
    return new;
}

static PyObject *element_form_spec_form_offset(const element_form_spec_t *self, PyObject *args, PyObject *kwds)
{
    long index, order_1, order_2;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "lll", (char *[4]){"", "order_1", "order_2", NULL}, &index, &order_1,
                                     &order_2))
        return NULL;
    if (index < 0 || index >= Py_SIZE(self))
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
    if (order_1 <= 0 || order_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders must be positive, but (%ld, %ld) were given", order_1, order_2);
        return NULL;
    }

    return PyLong_FromLong(element_form_offset(self, index, order_1, order_2));
}

PyDoc_STRVAR(element_form_spec_form_order_docstr,
             "form_offset(idx: typing.SupportsIndex, /, order_1: int, order_2: int) -> int\n"
             "Get the offset of the form in the element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : typing.SupportsIndex\n"
             "    Index of the form.\n"
             "\n"
             "order_1 : int\n"
             "    Order of the element in the first dimension.\n"
             "\n"
             "order_2 : int\n"
             "    Order of the element in the second dimension.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int\n"
             "    Offset of degrees of freedom of the differential form.\n");

static PyObject *element_form_spec_form_size(const element_form_spec_t *const self, PyObject *args, PyObject *kwds)
{
    long index, order_1, order_2;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "lll", (char *[4]){"", "order_1", "order_2", NULL}, &index, &order_1,
                                     &order_2))
        return NULL;
    if (index < 0 || index >= Py_SIZE(self))
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
    if (order_1 <= 0 || order_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders must be positive, but (%ld, %ld) were given", order_1, order_2);
        return NULL;
    }
    return PyLong_FromLong(form_degrees_of_freedom_count(self->forms[index].order, order_1, order_2));
}
PyDoc_STRVAR(element_form_spec_form_size_docstr,
             "form_size(idx: typing.SupportsIndex, /, order_1: int, order_2: int) -> int\n"
             "Get the number of degrees of freedom of the form in the element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : typing.SupportsIndex\n"
             "    Index of the form.\n"
             "\n"
             "order_1 : int\n"
             "    Order of the element in the first dimension.\n"
             "\n"
             "order_2 : int\n"
             "    Order of the element in the second dimension.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int\n"
             "    Number of degrees of freedom of the differential form.\n");

static PyObject *element_form_spec_form_total_size(const element_form_spec_t *const self, PyObject *args,
                                                   PyObject *kwds)
{
    long order_1, order_2;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ll", (char *[3]){"order_1", "order_2", NULL}, &order_1, &order_2))
        return NULL;
    if (order_1 <= 0 || order_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders must be positive, but (%ld, %ld) were given", order_1, order_2);
        return NULL;
    }
    // unsigned size = 0;
    // for (unsigned i = 0; i < Py_SIZE(self); ++i)
    // {
    //     size += form_degrees_of_freedom_count(self->forms[i].order, order_1, order_2);
    // }

    return PyLong_FromLong(element_form_specs_total_count(self, order_1, order_2));
}

PyDoc_STRVAR(element_form_spec_size_total_docstr,
             "total_size(order_1: int, order_2: int) -> int\n"
             "Get the total number of degrees of freedom of the forms.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "idx : typing.SupportsIndex\n"
             "    Index of the form.\n"
             "\n"
             "order_1 : int\n"
             "    Order of the element in the first dimension.\n"
             "\n"
             "order_2 : int\n"
             "    Order of the element in the second dimension.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int\n"
             "    Total number of degrees of freedom of all differential forms.\n");

static PyObject *element_form_spec_form_orders(const element_form_spec_t *self, PyObject *args, PyObject *kwds)
{
    long order_1, order_2;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ll", (char *[3]){"order_1", "order_2", NULL}, &order_1, &order_2))
        return NULL;
    if (order_1 <= 0 || order_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders must be positive, but (%ld, %ld) were given", order_1, order_2);
        return NULL;
    }
    PyObject *res = PyTuple_New(Py_SIZE(self) + 1);
    if (!res)
        return NULL;
    unsigned offsets = 0;
    for (unsigned i = 0; i < Py_SIZE(self); ++i)
    {
        const unsigned size = form_degrees_of_freedom_count(self->forms[i].order, order_1, order_2);
        PyObject *const v = PyLong_FromLong(offsets);
        if (!v)
        {
            Py_DECREF(res);
            return NULL;
        }
        PyTuple_SET_ITEM(res, i, v);
        offsets += size;
    }
    PyObject *const v = PyLong_FromLong(offsets);
    if (!v)
    {
        Py_DECREF(res);
        return NULL;
    }
    PyTuple_SET_ITEM(res, Py_SIZE(self), v);
    return res;
}

PyDoc_STRVAR(element_form_spec_form_orders_docstr,
             "form_offsets(order_1: int, order_2: int) -> tuple[int, ...]\n"
             "Get the offsets of all forms in the element.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "order_1 : int\n"
             "    Order of the element in the first dimension.\n"
             "\n"
             "order_2 : int\n"
             "    Order of the element in the second dimension.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "tuple of int\n"
             "    Offsets of degrees of freedom for all differential forms, with an extra\n"
             "    entry at the end, which is the count of all degrees of freedom.\n");

static PyObject *element_form_spec_form_sizes(const element_form_spec_t *this, PyObject *args, PyObject *kwds)
{
    long order_1, order_2;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ll", (char *[3]){"order_1", "order_2", NULL}, &order_1, &order_2))
        return NULL;
    if (order_1 <= 0 || order_2 <= 0)
    {
        PyErr_Format(PyExc_ValueError, "Orders must be positive, but (%ld, %ld) were given", order_1, order_2);
        return NULL;
    }
    PyObject *res = PyTuple_New(Py_SIZE(this));
    if (!res)
        return NULL;

    for (Py_ssize_t i = 0; i < Py_SIZE(this); ++i)
    {
        PyObject *val = PyLong_FromLong(form_degrees_of_freedom_count(this->forms[i].order, order_1, order_2));
        if (!val)
        {
            Py_DECREF(res);
            return NULL;
        }
        PyTuple_SET_ITEM(res, i, val);
    }
    return res;
}

PyDoc_STRVAR(element_form_spec_form_sizes_docstr, "form_sizes(order_1: int, order_2: int) -> tuple[int, ...]\n"
                                                  "Get the number of degrees of freedom for each form in the element.\n"
                                                  "\n"
                                                  "Parameters\n"
                                                  "----------\n"
                                                  "order_1 : int\n"
                                                  "    Order of the element in the first dimension.\n"
                                                  "\n"
                                                  "order_2 : int\n"
                                                  "    Order of the element in the second dimension.\n"
                                                  "\n"
                                                  "Returns\n"
                                                  "-------\n"
                                                  "tuple of int\n"
                                                  "    Number of degrees of freedom for each differential form.\n");

static PyObject *element_form_spec_get_index(const element_form_spec_t *this, PyObject *arg)
{
    if (!PyTuple_Check(arg) || PyTuple_GET_SIZE(arg) != 2)
    {
        PyErr_Format(PyExc_TypeError, "Expected a tuple of (name, order) but got %R", arg);
        return NULL;
    }

    const char *name_cstr = PyUnicode_AsUTF8(PyTuple_GET_ITEM(arg, 0));
    if (!name_cstr)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a string for the form name");
        return NULL;
    }
    if (strlen(name_cstr) > MAXIMUM_FORM_NAME_LENGTH)
    {
        PyErr_Format(PyExc_ValueError, "Form name longer than %d characters", MAXIMUM_FORM_NAME_LENGTH);
        return NULL;
    }
    const form_order_t order_val = form_order_from_object(PyTuple_GET_ITEM(arg, 1));
    if (order_val == FORM_ORDER_UNKNOWN)
    {
        raise_exception_from_current(PyExc_ValueError, "Failed converting form order to a number.");
        return NULL;
    }
    for (unsigned i = 0; i < Py_SIZE(this); ++i)
    {
        const form_spec_t *const spec = this->forms + i;
        if (spec->order == order_val && strcmp(spec->name, name_cstr) == 0)
        {
            return PyLong_FromLong(i);
        }
    }

    PyErr_Format(PyExc_ValueError, "Form with name %s and order %u not found", name_cstr, order_val);
    return NULL;
}

PyDoc_STRVAR(element_form_spec_get_index_docstr,
             "index(value: tuple[str, int], /) -> int\n"
             "Return the index of the form with the given label and order in the specs.\n"
             "\n"
             "Parameters\n"
             "----------\n"
             "value : tuple of (str, int)\n"
             "    Label and index of the form.\n"
             "\n"
             "Returns\n"
             "-------\n"
             "int\n"
             "    Index of the form in the specification.\n");

static PyMethodDef element_form_spec_methods[] = {
    {
        .ml_name = "form_offset",
        .ml_meth = (void *)element_form_spec_form_offset,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = element_form_spec_form_order_docstr,
    },
    {
        .ml_name = "form_size",
        .ml_meth = (void *)element_form_spec_form_size,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = element_form_spec_form_size_docstr,
    },
    {
        .ml_name = "total_size",
        .ml_meth = (void *)element_form_spec_form_total_size,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = element_form_spec_size_total_docstr,
    },
    {
        .ml_name = "form_offsets",
        .ml_meth = (void *)element_form_spec_form_orders,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = element_form_spec_form_orders_docstr,
    },
    {
        .ml_name = "form_sizes",
        .ml_meth = (void *)element_form_spec_form_sizes,
        .ml_flags = METH_VARARGS | METH_KEYWORDS,
        .ml_doc = element_form_spec_form_sizes_docstr,
    },
    {
        .ml_name = "index",
        .ml_meth = (void *)element_form_spec_get_index,
        .ml_flags = METH_O,
        .ml_doc = element_form_spec_get_index_docstr,
    },
    {},
};

PyDoc_STRVAR(element_form_spec_docstr, "_ElementFormSpecification(*specs: tuple[str, int])\n"
                                       "Specifications of forms defined on an element.\n"
                                       "\n"
                                       "Parameters\n"
                                       "----------\n"
                                       "*specs : tuple of (str, int)\n"
                                       "    Specifications for differential forms on the element. Each label must be\n"
                                       "    unique and order have a valid value which is in\n"
                                       "    :class:`mfv2d.kform.UnknownFormOrder`.\n");

PyTypeObject element_form_spec_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "mfv2d._mfv2d._ElementFormSpecification",
    .tp_basicsize = sizeof(element_form_spec_t),
    .tp_itemsize = sizeof(form_spec_t),
    .tp_dealloc = (destructor)element_form_spec_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_doc = element_form_spec_docstr,
    .tp_new = element_form_spec_new,
    .tp_getset = element_form_spec_getset,
    .tp_as_sequence = &element_form_spec_sequence_methods,
    .tp_repr = (reprfunc)element_form_spec_repr,
    .tp_str = (reprfunc)element_form_spec_str,
    .tp_methods = element_form_spec_methods,
    .tp_iter = (getiterfunc)element_form_spec_iter,
};
