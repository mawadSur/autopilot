# delete.py

import boto3

sagemaker = boto3.client('sagemaker')

def delete_trial_components():
    """
    Finds and deletes SageMaker Trial Components.
    """
    print("Starting trial component cleanup...")
    paginator = sagemaker.get_paginator('list_trial_components')
    for page in paginator.paginate():
        for summary in page.get('TrialComponentSummaries', []):
            name = summary['TrialComponentName']
            arn = summary['TrialComponentArn']
            print(f"\nProcessing Trial Component: {name}")

            associations = sagemaker.list_associations(SourceArn=arn)
            for assoc in associations.get('AssociationSummaries', []):
                trial_name = assoc['SourceName']
                print(f"  Disassociating from trial: {trial_name}")
                sagemaker.disassociate_trial_component(
                    TrialComponentName=name,
                    TrialName=trial_name
                )
            try:
                sagemaker.delete_trial_component(TrialComponentName=name)
                print(f"  Deleted trial component: {name}")
            except Exception as e:
                print(f"  ERROR deleting {name}: {str(e)}")


def delete_endpoints():
    """
    Finds and deletes all SageMaker Endpoints that are not already deleting.
    """
    print("\n🗑️ Starting endpoint cleanup...")

    paginator = sagemaker.get_paginator('list_endpoints')
    endpoints_found = 0
    
    # Iterate through all pages of endpoints
    for page in paginator.paginate():
        for ep in page.get('Endpoints', []):
            endpoints_found += 1
            ep_name = ep['EndpointName']
            ep_status = ep['EndpointStatus']
            print(f"Found endpoint: {ep_name} (status: {ep_status})")

            # Only attempt to delete if it's not already in the 'Deleting' state
            if ep_status != 'Deleting':
                try:
                    # Delete the associated endpoint config first
                    ep_config_name = sagemaker.describe_endpoint(EndpointName=ep_name)['EndpointConfigName']
                    sagemaker.delete_endpoint_config(EndpointConfigName=ep_config_name)
                    print(f"  - Deleted endpoint config: {ep_config_name}")

                    # Now delete the endpoint
                    sagemaker.delete_endpoint(EndpointName=ep_name)
                    print(f"  - Deleting endpoint: {ep_name}")
                except Exception as e:
                    print(f"  ❌ ERROR deleting {ep_name}: {str(e)}")
    
    if endpoints_found == 0:
        print("No active endpoints found.")


if __name__ == "__main__":
    # The script will now clean up both trials and endpoints
    delete_trial_components()
    delete_endpoints()
    print("\n✅ SageMaker cleanup completed.")