using Python.Runtime;

PythonEngine.Initialize();

using (Py.GIL())
{
    dynamic math = Py.Import("math");
    Console.WriteLine(math.sqrt(25));
}