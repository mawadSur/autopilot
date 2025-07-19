import boto3

sagemaker = boto3.client('sagemaker')

def delete_trial_components():
    print("Starting trial component cleanup...")

    paginator = sagemaker.get_paginator('list_trial_components')
    for page in paginator.paginate():
        for summary in page.get('TrialComponentSummaries', []):
            name = summary['TrialComponentName']
            arn = summary['TrialComponentArn']
            print(f"\nProcessing Trial Component: {name}")

            # Get associated trials
            associations = sagemaker.list_associations(SourceArn=arn)
            for assoc in associations.get('AssociationSummaries', []):
                print(assoc['SourceName'])
                trial_name = assoc['SourceName']
                print(f"  Disassociating from trial: {trial_name}")
                sagemaker.disassociate_trial_component(
                    TrialComponentName=name,
                    TrialName=trial_name
                )

            # Delete trial component
            try:
                sagemaker.delete_trial_component(TrialComponentName=name)
                print(f"  Deleted trial component: {name}")
            except Exception as e:
                print(f"  ERROR deleting {name}: {str(e)}")

def delete_endpoints():
    print("\nStarting endpoint cleanup...")

    paginator = sagemaker.get_paginator('list_endpoints')
    for page in paginator.paginate():
        for ep in page.get('Endpoints', []):
            ep_name = ep['EndpointName']
            ep_status = ep['EndpointStatus']
            print(f"Found endpoint: {ep_name} (status: {ep_status})")

            if ep_status != 'Deleting':
                try:
                    sagemaker.delete_endpoint(EndpointName=ep_name)
                    print(f"  Deleted endpoint: {ep_name}")
                except Exception as e:
                    print(f"  ERROR deleting endpoint {ep_name}: {str(e)}")

if __name__ == "__main__":
    delete_trial_components()
    delete_endpoints()
    print("\n✅ SageMaker cleanup completed.")
