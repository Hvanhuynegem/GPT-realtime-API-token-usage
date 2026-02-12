using Thesis.Experiments;

class Program
{
    static async Task Main(string[] args)
    {
        await ExperimentPlan.RunAsync();
    }
}


// class Program
// {
//     static async Task Main(string[] args)
//     {
//         await DatasetExperiment.RunAsync();
//     }
// }

// class Program
// {
//     static void Main(string[] args)
//     {
//         PreprocessingTechniquesTestbench.Run();
//     }
// }

// class Program
// {
//     static void Main(string[] args)
//     {
//         GazeRoiExperiment.Run();
//     }
// }

